import argparse
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from tensorboardX import SummaryWriter

from analysis import classification
from datasets import sst2, headerless_tsv
from loss import ClassificationLossCompute
from model_pytorch import DoubleHeadModel, load_openai_pretrained_model
from opt import OpenAIAdam
from text_utils import TextEncoder
from train import predict, iter_apply, transform_classification
from utils import (encode_dataset, iter_data,
                   make_path, ResultLogger)


def log(save_dir, desc):
    global best_score
    va_logits, va_cost = iter_apply(vaX, vaM, vaY,
                                    dh_model, compute_loss_fct, n_batch_train, device)
    va_cost = va_cost / n_valid
    va_acc = accuracy_score(vaY, np.argmax(va_logits, 1)) * 100.
    logger.log(n_epochs=n_epochs, n_updates=n_updates, va_cost=va_cost, va_acc=va_acc)
    tensorboard_logger.add_scalar(
        'val_loss', va_cost, n_updates)
    tensorboard_logger.add_scalar(
        'val_accuracy', va_acc, n_updates)

    score = va_acc
    if score > best_score:
        best_score = score
        path = os.path.join(save_dir, desc, 'best_params')
        torch.save(dh_model.state_dict(), make_path(path))


def run_epoch(update_internal):
    for xmb, mmb, ymb in iter_data(*shuffle(trX, trM, trY, random_state=np.random),
                                   n_batch=n_batch_train, truncate=True, verbose=True):
        global n_updates
        dh_model.train()
        XMB = torch.tensor(xmb, dtype=torch.long).to(device)
        YMB = torch.tensor(ymb, dtype=torch.long).to(device)
        MMB = torch.tensor(mmb).to(device)
        lm_logits, clf_logits = dh_model(XMB)
        train_loss = compute_loss_fct(XMB, YMB, MMB, clf_logits, lm_logits)
        n_updates += 1
        tensorboard_logger.add_scalar(
            'train_loss', train_loss, n_updates)
        if n_updates % update_internal == 0:
            log(save_dir, desc)


argmax = lambda x: np.argmax(x, 1)

preprocess_fns = defaultdict(lambda: headerless_tsv, sst2=sst2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--submission_dir', type=str, default='submission/')
    parser.add_argument('--skip_preprocess', action='store_true')
    parser.add_argument('--update_interval', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--n_batch', type=int, default=8)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_ctx', type=int, default=512)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default='model/vocab_40000.bpe')
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Constants
    n_ctx = args.n_ctx
    desc = args.dataset
    save_dir = os.path.join(args.save_dir, desc)
    data_dir = os.path.join(args.data_dir, desc)
    log_dir = os.path.join(args.log_dir, desc)
    submission_dir = args.submission_dir

    dataset = args.dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)

    log_file = os.path.join(log_dir, '{}.jsonl'.format(dataset))
    logger = ResultLogger(path=log_file, **args.__dict__)
    text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)

    print("Encoding dataset...")
    ((trX, trY),
     (vaX, vaY),
     (teX, teY)) = encode_dataset(*preprocess_fns[dataset](data_dir),
                                  encoder=text_encoder,
                                  skip_preprocess=args.skip_preprocess)
    encoder['_start_'] = len(encoder)
    encoder['_classify_'] = len(encoder)
    clf_token = encoder['_classify_']
    n_special = 2
    max_len = n_ctx - n_special
    n_ctx = min(max([len(x[:max_len]) for x in trX] +
                    [len(x[:max_len]) for x in vaX] +
                    [len(x[:max_len]) for x in teX]) + n_special, n_ctx)
    vocab = n_vocab + n_special + n_ctx


    def transform(X):
        return transform_classification(X, max_len, encoder['_start_'], clf_token,
                                        n_vocab, n_special, n_ctx)


    trX, trM = transform(trX)
    vaX, vaM = transform(vaX)
    teX, teM = transform(teX)

    n_class = len(set(trY) | set(vaY) | set(teY))

    dh_model = DoubleHeadModel(args, clf_token, ['classification', n_class], vocab, n_ctx)

    n_train = len(trY)
    n_valid = len(vaY)

    n_batch_train = args.n_batch * max(n_gpu, 1)
    n_updates_total = (n_train // n_batch_train) * args.n_iter

    criterion = nn.CrossEntropyLoss(reduce=False)
    model_opt = OpenAIAdam(dh_model.parameters(),
                           lr=args.lr,
                           schedule=args.lr_schedule,
                           warmup=args.lr_warmup,
                           t_total=n_updates_total,
                           b1=args.b1,
                           b2=args.b2,
                           e=args.e,
                           l2=args.l2,
                           vector_l2=args.vector_l2,
                           max_grad_norm=args.max_grad_norm)
    compute_loss_fct = ClassificationLossCompute(criterion,
                                                 criterion,
                                                 args.lm_coef,
                                                 model_opt)
    load_openai_pretrained_model(dh_model.transformer, n_ctx=n_ctx, n_special=n_special, n_transfer=args.n_transfer)

    dh_model.to(device)
    dh_model = nn.DataParallel(dh_model)

    n_updates = 0
    n_epochs = 0
    path = os.path.join(save_dir, 'best_params')
    torch.save(dh_model.state_dict(), make_path(path))

    best_score = 0
    tensorboard_logger = SummaryWriter(log_dir)
    for i in range(args.n_iter):
        print("running epoch", i)
        run_epoch(args.update_interval)
        n_epochs += 1
        log(save_dir, desc)

    path = os.path.join(save_dir, 'best_params')
    dh_model.load_state_dict(torch.load(path))
    predict_file = '{}.tsv'.format(dataset)
    predict(X=teX,
            submission_dir=args.submission_dir,
            filename=predict_file,
            pred_fn=argmax,
            label_decoder=None,
            dh_model=dh_model,
            n_batch_train=n_batch_train,
            device=device)
    classification(dataset,
                   teY,
                   os.path.join(args.submission_dir, predict_file),
                   log_file)

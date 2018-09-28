import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from analysis import rocstories as rocstories_analysis
from datasets import sst2
from loss import ClassificationLossCompute
from model_pytorch import DoubleHeadModel, load_openai_pretrained_model
from opt import OpenAIAdam
from text_utils import TextEncoder
from train import predict, iter_apply, transform_classification
from utils import (encode_dataset, iter_data,
                   ResultLogger, make_path)


def log(save_dir, desc):
    global best_score
    print("Logging")
    tr_logits, tr_cost = iter_apply(trX[:n_valid], trM[:n_valid], trY[:n_valid],
                                    dh_model, compute_loss_fct, n_batch_train, device)
    va_logits, va_cost = iter_apply(vaX, vaM, vaY,
                                    dh_model, compute_loss_fct, n_batch_train, device)
    tr_cost = tr_cost / len(trY[:n_valid])
    va_cost = va_cost / n_valid
    tr_acc = accuracy_score(trY[:n_valid], np.argmax(tr_logits, 1)) * 100.
    va_acc = accuracy_score(vaY, np.argmax(va_logits, 1)) * 100.
    logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=tr_cost, va_cost=va_cost, tr_acc=tr_acc, va_acc=va_acc)
    print('%d %d %.3f %.3f %.2f %.2f' % (n_epochs, n_updates, tr_cost, va_cost, tr_acc, va_acc))
    if submit:
        score = va_acc
        if score > best_score:
            best_score = score
            path = os.path.join(save_dir, desc, 'best_params')
            torch.save(dh_model.state_dict(), make_path(path))


def run_epoch():
    for xmb, mmb, ymb in iter_data(*shuffle(trX, trM, trY, random_state=np.random),
                                   n_batch=n_batch_train, truncate=True, verbose=True):
        global n_updates
        dh_model.train()
        XMB = torch.tensor(xmb, dtype=torch.long).to(device)
        YMB = torch.tensor(ymb, dtype=torch.long).to(device)
        MMB = torch.tensor(mmb).to(device)
        lm_logits, clf_logits = dh_model(XMB)
        compute_loss_fct(XMB, YMB, MMB, clf_logits, lm_logits)
        n_updates += 1
        if n_updates in [1000, 2000, 4000, 8000, 16000, 32000] and n_epochs == 0:
            log(save_dir, desc)


argmax = lambda x: np.argmax(x, 1)

preprocess_fns = {
    'sst2': sst2,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--submission_dir', type=str, default='submission/')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--analysis', action='store_true')
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
    submit = args.submit
    n_ctx = args.n_ctx
    save_dir = args.save_dir
    desc = args.dataset
    data_dir = args.data_dir
    log_dir = args.log_dir
    submission_dir = args.submission_dir

    dataset = args.dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)

    logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)
    text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)

    print("Encoding dataset...")
    ((trX, trY),
     (vaX, vaY),
     (teX, teY)) = encode_dataset(*preprocess_fns[dataset](data_dir), encoder=text_encoder)
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
    if submit:
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
    if submit:
        path = os.path.join(save_dir, desc, 'best_params')
        torch.save(dh_model.state_dict(), make_path(path))
    best_score = 0
    for i in range(args.n_iter):
        print("running epoch", i)
        run_epoch()
        n_epochs += 1
        log(save_dir, desc)
    if submit:
        path = os.path.join(save_dir, desc, 'best_params')
        dh_model.load_state_dict(torch.load(path))
        predict(teX, teM, args.submission_dir, '{}.tsv'.format(dataset), argmax, None)
        if args.analysis:
            rocstories_analysis(data_dir, os.path.join(args.submission_dir, '{}.tsv'.format(dataset)),
                                os.path.join(log_dir, '{}.jsonl'.format(dataset)))

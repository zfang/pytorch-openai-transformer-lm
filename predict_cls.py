import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from datasets import load_headerless_tsv
from model_pytorch import DoubleHeadModel, dotdict
from text_utils import TextEncoder
from train import transform_classification, predict
from utils import encode_dataset, softmax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', required=True)
    parser.add_argument('-o', '--output_file', required=True)
    parser.add_argument('--n_batch', type=int, default=8)
    parser.add_argument('--skip_preprocess', action='store_true')
    parser.add_argument('--sentence_pair', action='store_true')
    parser.add_argument('--force_delimiter', action='store_true')
    parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default='model/vocab_40000.bpe')
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--mc_dropout_iter', type=int, default=0)
    args = parser.parse_args()

    meta = json.load(open(os.path.join(args.model_dir, 'meta.json'), 'r', encoding='utf8'))

    text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)

    encoder['_start_'] = len(encoder)
    if args.sentence_pair or args.force_delimiter:
        encoder['_delimiter_'] = len(encoder)
    encoder['_classify_'] = len(encoder)
    clf_token = encoder['_classify_']
    n_ctx = meta['dh_model']['n_ctx']
    max_len = meta['encoder']['max_len']
    n_special = 2

    texts, labels = load_headerless_tsv(args.input_file, sentence_pair=args.sentence_pair)
    ((X, Y),) = encode_dataset(*[(texts, labels)],
                               encoder=text_encoder,
                               skip_preprocess=args.skip_preprocess)

    X, M = transform_classification(X, max_len, encoder['_start_'], clf_token,
                                    n_vocab, n_special, n_ctx, encoder.get('_delimiter_'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    n_batch_train = args.n_batch * max(n_gpu, 1)

    meta['dh_model']['cfg'] = dotdict(meta['dh_model']['cfg'])
    dh_model = DoubleHeadModel(**meta['dh_model'])
    dh_model.to(device)
    dh_model = torch.nn.DataParallel(dh_model)
    path = os.path.join(args.model_dir, 'best_params')
    if device == torch.device('cpu'):
        map_location = lambda storage, loc: storage
    else:
        map_location = None

    dh_model.load_state_dict(torch.load(path, map_location=map_location))
    prediction_output = predict(X=X,
                                submission_dir=None,
                                filename=None,
                                pred_fn=lambda x: x,
                                label_decoder=None,
                                dh_model=dh_model,
                                n_batch_train=n_batch_train,
                                device=device)

    predictions = np.argmax(prediction_output, axis=1)
    if type(texts) is tuple:
        df = pd.DataFrame({'question': texts[0], 'text': texts[1], 'label': labels, 'prediction': predictions})
    else:
        df = pd.DataFrame({'text': texts, 'label': labels, 'prediction': predictions})
    df.to_csv(args.output_file,
              index=False,
              sep='\t',
              header=False,
              columns=['text', 'label', 'prediction'],
              float_format='%.0f')

    accuracy = accuracy_score(Y, predictions) * 100.
    print('Accuracy: {}%'.format(accuracy))

    basename = os.path.splitext(args.output_file)[0]

    prediction_output_file = basename + '_output.npy'
    np.savetxt(prediction_output_file, prediction_output)
    prediction_probs = softmax(prediction_output)
    prediction_probs_file = basename + '_probs.npy'
    np.savetxt(prediction_probs_file, prediction_probs)

    mc_dropout_prediction_output = []
    for _ in tqdm(range(args.mc_dropout_iter)):
        prediction_output = predict(X=X,
                                    submission_dir=None,
                                    filename=None,
                                    pred_fn=lambda x: x,
                                    label_decoder=None,
                                    dh_model=dh_model,
                                    n_batch_train=n_batch_train,
                                    device=device,
                                    enable_dropout=True)
        mc_dropout_prediction_output.append(prediction_output)

    if mc_dropout_prediction_output:
        mc_dropout_prediction_output = np.asarray(mc_dropout_prediction_output)
        mc_dropout_prediction_probs = np.zeros(mc_dropout_prediction_output.shape)
        for i in range(mc_dropout_prediction_output.shape[0]):
            mc_dropout_prediction_probs[i, ...] = softmax(mc_dropout_prediction_output[i, ...])

        transpose_dims = (2, 1, 0)
        mc_dropout_prediction_output = mc_dropout_prediction_output.transpose(transpose_dims)
        mc_dropout_prediction_probs = mc_dropout_prediction_probs.transpose(transpose_dims)
        for i in range(mc_dropout_prediction_output.shape[0]):
            prediction_output_file = '{}_class{}_{}'.format(basename, i, 'output.npy')
            np.savetxt(prediction_output_file, mc_dropout_prediction_output[i, ...])
            prediction_probs_file = '{}_class{}_{}'.format(basename, i, 'probs.npy')
            np.savetxt(prediction_probs_file, mc_dropout_prediction_probs[i, ...])


if __name__ == '__main__':
    main()

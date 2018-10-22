import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

from datasets import load_headerless_tsv
from model_pytorch import DoubleHeadModel, dotdict
from text_utils import TextEncoder
from train import transform_classification, predict
from utils import encode_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', required=True)
    parser.add_argument('-o', '--output_file', required=True)
    parser.add_argument('--n_batch', type=int, default=8)
    parser.add_argument('--skip_preprocess', action='store_true')
    parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default='model/vocab_40000.bpe')
    parser.add_argument('--model_dir', required=True)
    args = parser.parse_args()

    meta = json.load(open(os.path.join(args.model_dir, 'meta.json'), 'r', encoding='utf8'))

    text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)

    encoder['_start_'] = len(encoder)
    encoder['_classify_'] = len(encoder)
    clf_token = encoder['_classify_']
    n_ctx = meta['dh_model']['n_ctx']
    max_len = meta['encoder']['max_len']
    n_special = 2

    texts, labels = load_headerless_tsv(args.input_file)
    ((X, Y),) = encode_dataset(*[(texts, labels)],
                               encoder=text_encoder,
                               skip_preprocess=args.skip_preprocess)

    X, M = transform_classification(X, max_len, encoder['_start_'], clf_token,
                                    n_vocab, n_special, n_ctx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    n_batch_train = args.n_batch * max(n_gpu, 1)

    meta['dh_model']['cfg'] = dotdict(meta['dh_model']['cfg'])
    dh_model = DoubleHeadModel(**meta['dh_model'])
    dh_model.to(device)
    dh_model = torch.nn.DataParallel(dh_model)
    path = os.path.join(args.model_dir, 'best_params')
    if device == 'cpu':
        map_location = lambda storage, loc: storage
    else:
        map_location = None

    dh_model.load_state_dict(torch.load(path, map_location=map_location))
    predictions = predict(X=X,
                          submission_dir='/tmp',
                          filename='predictions.tsv',
                          pred_fn=lambda x: np.argmax(x, 1),
                          label_decoder=None,
                          dh_model=dh_model,
                          n_batch_train=n_batch_train,
                          device=device)

    df = pd.DataFrame({'text': texts, 'label': labels, 'prediction': predictions})
    df.to_csv(args.output_file,
              index=False,
              sep='\t',
              header=False,
              columns=['text', 'label', 'prediction'],
              float_format='%.0f')

    accuracy = accuracy_score(Y, predictions) * 100.
    print('Accuracy: {}%'.format(accuracy))


if __name__ == '__main__':
    main()

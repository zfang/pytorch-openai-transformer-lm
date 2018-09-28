import os

import numpy as np
import torch

from utils import iter_data


def transform_classification(X, max_len, start, clf_token, n_vocab, n_special, n_ctx):
    n_batch = len(X)
    xmb = np.zeros((n_batch, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, n_ctx), dtype=np.float32)
    for i, x in enumerate(X):
        x_input = [start] + x[:max_len] + [clf_token]
        l = len(x_input)
        xmb[i, :l, 0] = x_input
        mmb[i, :l] = 1
    xmb[:, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    return xmb, mmb


def iter_apply(Xs, Ms, Ys, dh_model, compute_loss_fct, n_batch_train, device):
    logits = []
    cost = 0
    with torch.no_grad():
        dh_model.eval()
        for xmb, mmb, ymb in iter_data(Xs, Ms, Ys, n_batch=n_batch_train, truncate=False, verbose=True):
            n = len(xmb)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            YMB = torch.tensor(ymb, dtype=torch.long).to(device)
            MMB = torch.tensor(mmb).to(device)
            _, clf_logits = dh_model(XMB)
            clf_logits *= n
            clf_losses = compute_loss_fct(XMB, YMB, MMB, clf_logits, only_return_losses=True)
            clf_losses *= n
            logits.append(clf_logits.to("cpu").numpy())
            cost += clf_losses.sum().item()
        logits = np.concatenate(logits, 0)
    return logits, cost


def iter_predict(Xs, dh_model, n_batch_train, device):
    logits = []
    with torch.no_grad():
        dh_model.eval()
        for xmb in iter_data(Xs, n_batch=n_batch_train, truncate=False, verbose=True):
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            _, clf_logits = dh_model(XMB)
            logits.append(clf_logits.to("cpu").numpy())
    logits = np.concatenate(logits, 0)
    return logits


def predict(X, submission_dir, filename, pred_fn, label_decoder, dh_model, n_batch_train, device):
    predictions = pred_fn(iter_predict(X, dh_model, n_batch_train, device))
    if label_decoder is not None:
        predictions = [label_decoder[prediction] for prediction in predictions]
    path = os.path.join(submission_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('{}\t{}\n'.format('index', 'prediction'))
        for i, prediction in enumerate(predictions):
            f.write('{}\t{}\n'.format(i, prediction))

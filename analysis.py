import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from datasets import _rocstories


def rocstories(data_dir, pred_path, log_path):
    preds = pd.read_csv(pred_path, delimiter='\t')['prediction'].values.tolist()
    _, _, _, labels = _rocstories(os.path.join(data_dir, 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'))
    test_accuracy = accuracy_score(labels, preds) * 100.
    logs = [json.loads(line) for line in open(log_path)][1:]
    best_validation_index = np.argmax([log['va_acc'] for log in logs])
    valid_accuracy = logs[best_validation_index]['va_acc']
    print('rocstories valid accuracy: %.2f' % (valid_accuracy))
    print('rocstories test accuracy:  %.2f' % (test_accuracy))


def classification(dataset, labels, pred_path, log_path):
    preds = pd.read_csv(pred_path, delimiter='\t')['prediction'].values.tolist()
    test_accuracy = accuracy_score(labels, preds) * 100.
    logs = [json.loads(line) for line in open(log_path)][1:]
    best_validation_index = np.argmax([log['va_acc'] for log in logs])
    valid_accuracy = logs[best_validation_index]['va_acc']
    print('%s valid accuracy: %.2f' % (dataset, valid_accuracy))
    print('%s test accuracy:  %.2f' % (dataset, test_accuracy))

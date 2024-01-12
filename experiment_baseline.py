import os
import sys
import logging
import numpy as np
import torch
torch.set_num_threads(2)
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from sklearn.model_selection import KFold, train_test_split
from sklearn import metrics
from collections import defaultdict

from rrl.utils import read_csv, DBEncoder
from rrl.models import RRL

DATA_DIR = './dataset'


def get_data(dataset, k=0):
    data_path = os.path.join(DATA_DIR, dataset + '.data')
    info_path = os.path.join(DATA_DIR, dataset + '.info')
    X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=True)

    db_enc = DBEncoder(f_df, discrete=False)
    db_enc.fit(X_df, y_df)

    X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    train_index, test_index = list(kf.split(X_df))[k]
    X_train = X[train_index]
    y_train = y[train_index, 0]
    X_test = X[test_index]
    y_test = y[test_index, 0]
    return X_train, y_train, X_test, y_test

def eval(y_label, y_pred):
    tp = np.sum((y_label == 1) * (y_pred == 1))
    fp = np.sum((y_label == 0) * (y_pred == 1))
    fn = np.sum((y_label == 1) * (y_pred == 0))
    tn = np.sum((y_label == 0) * (y_pred == 0))

    acc = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print(f"acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}")

from sklearn.linear_model import LogisticRegression
def train_and_test_lr(dataset):
    X_train, y_train, X_test, y_test = get_data(dataset)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    result = model.predict(X_test)
    eval(y_test, result)

class MyLogisticRegression():
    def __init__(self, lr=1e-1):
        self.w = None
        self.lr = lr

    def fit(self, X, y, n_iters=4000):
        n_feats = X.shape[1]
        bound = n_feats**-0.5
        self.w = np.random.uniform(-bound, bound, size=(n_feats,))

        for i in range(n_iters):
            y_pred = 1 / (1 + np.exp(-X.dot(self.w)))
            self.w -= self.lr * -(y-y_pred).dot(X)
    
    def predict(self, X):
        y_pred = 1 / (1 + np.exp(-X.dot(self.w)))
        return (y_pred >= 0.5).astype(np.int32)
def train_and_test_lr_manual(dataset):
    X_train, y_train, X_test, y_test = get_data(dataset)

    model = MyLogisticRegression()
    model.fit(X_train, y_train)
    result = model.predict(X_test)
    eval(y_test, result)

from sklearn.naive_bayes import GaussianNB
def train_and_test_nb(dataset):
    X_train, y_train, X_test, y_test = get_data(dataset)

    model = GaussianNB()
    model.fit(X_train, y_train)
    result = model.predict(X_test)
    eval(y_test, result)

from sklearn.tree import DecisionTreeClassifier
def train_and_test_dt(dataset):
    X_train, y_train, X_test, y_test = get_data(dataset)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    result = model.predict(X_test)
    eval(y_test, result)

from sklearn.svm import SVC
def train_and_test_svm(dataset):
    X_train, y_train, X_test, y_test = get_data(dataset)

    model = SVC()
    model.fit(X_train, y_train)
    result = model.predict(X_test)
    eval(y_test, result)

if __name__ == '__main__':
    mode = sys.argv[1]
    dataset = 'bank'
    
    assert mode in ['lr', 'lr_manual', 'nb', 'dt', 'svm'], "invalid mode!!"
    if mode == 'lr':
        train_and_test_lr(dataset)
    elif mode == 'lr_manual':
        train_and_test_lr_manual(dataset)
    elif mode == 'nb':
        train_and_test_nb(dataset)
    elif mode == 'dt':
        train_and_test_dt(dataset)
    elif mode =='svm':
        train_and_test_svm(dataset)
    else:
        raise NotImplementedError
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
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    return X_train, y_train, X_test, y_test

from sklearn import tree
def train_and_test_decision_tree_regression(dataset):
    X_train, y_train, X_test, y_test = get_data(dataset)

    model = tree.DecisionTreeRegressor()
    model.fit(X_train, y_train)
    result = model.predict(X_test)
    
    # add metrics calculation
    print('RMSE:', np.sqrt(np.mean(np.power(y_test - result, 2))))
    print('MAE:', np.mean(np.abs(y_test - result)))
    breakpoint()
    return

from sklearn.linear_model import Ridge
def train_and_test_ridge_regression(dataset):
    X_train, y_train, X_test, y_test = get_data(dataset)

    model = Ridge(alpha=1)
    model.fit(X_train, y_train)
    result = model.predict(X_test)
    
    # add metrics calculation
    print('RMSE:', np.sqrt(np.mean(np.power(y_test - result, 2))))
    print('MAE:', np.mean(np.abs(y_test - result)))
    return


class MyRidge:
    def __init__(self, alpha=1, n_iters=1000, lr=1e-5):
        self.alpha = alpha
        self.n_iters = n_iters
        self.lr = lr

    def fit(self, X, y):
        X_ = np.insert(X, 0, 1, axis=1)
        
        # initialize
        bound = (X_.shape[1])**-0.5
        self.w = np.random.uniform(-bound, bound, (X_.shape[1],))

        # regression
        for i in range(self.n_iters):
            y_pred = X_.dot(self.w)
            self.w -= self.lr * ((y_pred - y).dot(X_) + self.alpha * self.w)
    
    def predict(self, X):
        return np.insert(X, 0, 1, axis=1).dot(self.w)
    

def train_and_test_ridge_regression_raw(dataset):
    X_train, y_train, X_test, y_test = get_data(dataset)

    log_y_train = np.log(y_train)
    log_y_test = np.log(y_test)

    model = MyRidge(alpha=1)
    model.fit(X_train, y_train)
    result = model.predict(X_test)

    # model.fit(X_train, log_y_train)
    # log_result = model.predict(X_test)
    # print('LOG RMSE:', np.sqrt(np.mean(np.power(log_y_test - log_result, 2))))
    # print('LOG MAE:', np.mean(np.abs(log_y_test - log_result)))
    
    # add metrics calculation
    print('RMSE:', np.sqrt(np.mean(np.power(y_test - result, 2))))
    print('MAE:', np.mean(np.abs(y_test - result)))
    print('R2:', 1 - np.sum(np.power(y_test - result, 2)) / np.sum(np.power(y_test - np.mean(y_test), 2)))
    return


if __name__ == '__main__':
    mode = sys.argv[1]
    dataset = sys.argv[2]

    assert mode in ['tree', 'ridge', 'ridge_raw'], 'Invalid mode!!'
    assert dataset in ['RedWineQuality', 'WineQuality', 'BostonHousing', 'OnlineNewsPopularity'], 'Invalid dataset!!'
    if mode == 'tree':
        train_and_test_decision_tree_regression(dataset)
    elif mode == 'ridge':
        train_and_test_ridge_regression(dataset)
    elif mode == 'ridge_raw':
        train_and_test_ridge_regression_raw(dataset)
    else:
        raise NotImplementedError
    
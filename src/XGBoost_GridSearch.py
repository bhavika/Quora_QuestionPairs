#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Check this gist for xgboost wrapper: https://gist.github.com/slaypni/b95cb69fd1c82ca4c2ff
# Author: Kazuaki Tanida
# Source: https://www.kaggle.com/tanitter/introducing-kaggle-scripts/grid-search-xgboost-with-scikit-learn

import sys
import math
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV
from XGB_Baseline import xgtrain_X, xgtest_X, X_train, y_train
from sklearn.metrics import log_loss
import subprocess
sys.path.append('xgboost/wrapper/')


class XGBoostClassifier():
    def __init__(self, num_boost_round=10, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'binary:logistic'})

    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        self.label2num = {label: i for i, label in enumerate(sorted(set(y)))}
        dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)

    def predict(self, X):
        num2label = {i: label for label, i in self.label2num.items()}
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return np.array([num2label[i] for i in y])

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)

    def score(self, X, y):
        Y = self.predict_proba(X)
        return 1 / log_loss(y, Y)

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self


def main():
    clf = XGBoostClassifier(
        eval_metric='logloss',
        nthread=4,
        silent=1,
    )
    parameters = {
        'num_boost_round': [100, 250, 500],
        'eta': [0.05, 0.1, 0.3],
        'max_depth': [6, 9, 12],
        'subsample': [0.9, 1.0],
        'colsample_bytree': [1.0],
    }
    clf = GridSearchCV(clf, parameters, n_jobs=1, cv=2)

    clf.fit(X_train, y_train)
    best_parameters, score, _ = max(clf.cv_results_, key=lambda x: x[1])
    print('score:', score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
    print('predicted:', clf.predict([[1, 1]]))

    subprocess.call(['speech-dispatcher'])  # start speech dispatcher
    subprocess.call(['spd-say', '"your process has finished"'])


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import xgboost
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import log_loss
import subprocess

sys.path.append('xgboost/wrapper/')

train = pd.read_pickle('../data/train.pkl')
test = pd.read_pickle('../data/test.pkl')

featureset1 = ['len_q1c', 'len_q2c', 'words_q1c', 'words_q2c', 'chars_q1c', 'chars_q2c', 'wordshare']
featureset2 = ['qratio', 'wratio', 'partial_ratio', 'partial_tokenset', 'tokenset', 'partial_tokensort']

features = featureset1 + featureset2
# features = ['wordshare']
print (features)

X_train, X_test, y_train, y_test = train_test_split(train[features], train['is_duplicate'], test_size=0.2)


def gridsearch():
    clf = xgboost.XGBRegressor(
        objective='binary:logistic',
        nthread=4,
        silent=False,
    )
    parameters = {
        'n_estimators': [50, 100, 250, 500],
        'learning_rate': [0.05, 0.1, 0.3],
        'max_depth': [4, 6, 9, 12, 24],
        'subsample': [0.9, 1.0],
        'colsample_bylevel': [1.0]
    }

    clf = GridSearchCV(clf, parameters, n_jobs=1, cv=4)

    clf.fit(X_train, y_train)

    print "Best parameters", clf.best_params_

    # print "CV results"
    # print clf.cv_results_

    print('predicted:', clf.predict([[1, 1]]))

    subprocess.call(['speech-dispatcher'])  # start speech dispatcher
    subprocess.call(['spd-say', '"your process has finished"'])


if __name__ == '__main__':
    gridsearch()
# coding: utf-8

import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss, make_scorer
import subprocess
import logging
from sklearn.model_selection import train_test_split
import xgboost as xgb

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

train = pd.read_pickle('../data/train.pkl')
test = pd.read_pickle('../data/test.pkl')


featureset1 = ['len_q1c', 'len_q2c', 'words_q1c', 'words_q2c', 'chars_q1c', 'chars_q2c', 'wordshare']
featureset2 = ['qratio', 'wratio', 'partial_ratio', 'partial_tokenset', 'tokenset', 'partial_tokensort']

features = featureset1 + featureset2

print (features)

X_train, X_test, y_train, y_test = train_test_split(train[features], train['is_duplicate'], test_size=0.2)

X_train = X_train.astype(float)
X_test = X_test.astype(float)
y_train = y_train.astype(float)

y_train = y_train.values
y_test = y_test.astype(float)

accuracy = make_scorer(log_loss)

xgtrain_X = xgb.DMatrix(X_train, label=y_train)
xgtest_X = xgb.DMatrix(X_test, label=y_test)

d_test = test[features]

d_test = xgb.DMatrix(d_test)

watchlist = [(xgtrain_X, 'train'), (xgtest_X,'test')]

params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4


def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0-preds)
    return grad, hess


baseline_xgb = xgb.train(params, xgtrain_X, evals=watchlist, verbose_eval=10)
y_pred = baseline_xgb.predict(d_test)

sub = pd.DataFrame()
sub['test_id'] = test['test_id']
sub['is_duplicate'] = y_pred
sub.to_csv('xgb_baseline.csv', index=False)


subprocess.call(['speech-dispatcher'])        #start speech dispatcher
subprocess.call(['spd-say', '"your process has finished"'])






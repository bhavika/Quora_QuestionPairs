# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import logging
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

train = pd.read_pickle('../data/train.pkl')
test = pd.read_pickle('../data/test.pkl')

featureset1 = ['len_q1c', 'len_q2c', 'words_q1c', 'words_q2c', 'chars_q1c', 'chars_q2c', 'wordshare']
featureset2 = ['qratio', 'wratio', 'partial_ratio', 'partial_tokenset', 'tokenset', 'partial_tokensort']
featureset3 = [ 'norm_wmd', 'wmd']

#features = featureset1 + featureset2 + featureset3
features = ['wordshare'] + featureset3
print (features)

X_train, X_test, y_train, y_test = train_test_split(train[features], train['is_duplicate'], test_size=0.2)

X_train = X_train.astype(float)
X_test = X_test.astype(float)
y_train = y_train.astype(float)
y_train = y_train.values
y_test = y_test.astype(float)


xgtrain_X = xgb.DMatrix(X_train, label=y_train)
xgtest_X = xgb.DMatrix(X_test, label=y_test)

d_test = test[features]
d_test = xgb.DMatrix(d_test)

watchlist = [(xgtrain_X, 'train'), (xgtest_X,'test')]

# params = {}
# params['objective'] = 'binary:logistic'
# params['eval_metric'] = 'logloss'
# params['eta'] = 0.02
# params['max_depth'] = 8

params = {'n_estimators': 500, 'subsample': 0.9, 'learning_rate': 0.05, 'max_depth': 9, 'colsample_bylevel': 1.0,
          'objective': 'binary:logistic', 'eval_metric': 'logloss'}

baseline_xgb = xgb.train(params, xgtrain_X, evals=watchlist,  num_boost_round=400, verbose_eval=10)
y_pred = baseline_xgb.predict(d_test)

sns.set(font_scale = 1.5)
xgb.plot_importance(baseline_xgb)
plt.show()

sub = pd.DataFrame()
sub['test_id'] = test['test_id']
sub['is_duplicate'] = y_pred
sub.to_csv('../sub/xgb_baseline6.csv', index=False)


subprocess.call(['speech-dispatcher'])        #start speech dispatcher
subprocess.call(['spd-say', '"your process has finished"'])





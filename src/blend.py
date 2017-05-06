import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from numpy import inf, where
from time import time

train = pd.read_pickle('../data/train.pkl')
test = pd.read_pickle('../data/test.pkl')

featureset1 = ['len_q1c', 'len_q2c', 'words_q1c', 'words_q2c', 'chars_q1c', 'chars_q2c', 'wordshare']
featureset2 = ['qratio', 'wratio', 'partial_ratio', 'partial_tokenset', 'tokenset', 'partial_tokensort']
featureset3 = [ 'norm_wmd', 'wmd']

features = featureset3 + ['wordshare']


def blend(X, y, test):
    np.random.seed(7)

    n_folds = 10
    verbose = True
    shuffle = False

    skf = StratifiedKFold(n_folds)
    skf_splits = skf.split(X, y)

    private = pd.read_pickle('../data/test.pkl')

    print skf_splits

    clfs = [RandomForestRegressor(n_estimators=100, n_jobs=-1, criterion='mse'),
            RandomForestRegressor(n_estimators=100, n_jobs=-1, criterion='mse'),
            ExtraTreesRegressor(n_estimators=100, n_jobs=-1, criterion='mse'),
            ExtraTreesRegressor(n_estimators=100, n_jobs=-1, criterion='mse'),
            GradientBoostingRegressor(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]

    print "Creating train and test sets for blending"
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((test.shape[0], len(clfs)))

    if shuffle:
        idx = np.random.permutation(y.size)
        x = X[idx]
        y = y[idx]

    for j, clf in enumerate(clfs):
        print j, clf
        # change X.shape to test.shape
        dataset_blend_test_j = np.zeros((test.shape[0], n_folds))

        for i, (tr, te) in enumerate(skf_splits):
            print "Fold ", i
            print "Train ", tr , "Test ", te
            X_train = X[tr]
            y_train = y[tr]
            X_test = X[te]
            y_test = y[te]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            dataset_blend_train[te, j] = y_predict
            dataset_blend_test_j[:, i] = clf.predict(test)
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

    print

    print "Blending"
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    y_predict = clf.predict_proba(dataset_blend_test)

    print "Linear stretch of predictions to [0, 1]"
    y_submission = (y_predict - y_predict.min())/(y_predict.max() - y_predict.min())

    # tmp = np.vstack([range(1, len(y_submission)), y_submission]).T

    print "Saving "

    # sub = pd.DataFrame()
    # sub['test_id'] = private['test_id']
    # sub['is_duplicate'] = y_submission
    # sub.to_csv('../sub/blended1.csv', index=False)

    np.savetxt(fname='blended3.txt', X=y_submission, fmt="%d,%0.9f", header='test_id,is_duplicate',comments='')

y_train = np.array(train['is_duplicate'])
X_train = np.array(train[features])
X_test = np.array(test[features])

# print np.all(np.isfinite(X_train))
# print np.all(np.isfinite(X_test))
# print np.all(np.isfinite(y_train))

X_train[where(np.isinf(X_train))] = 0
X_test[where(np.isinf(X_test))] = 0

start_time = time()
blend(X_train, y_train, X_test)

print "Elapsed time: ", time() - start_time
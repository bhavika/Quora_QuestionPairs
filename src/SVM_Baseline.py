
# coding: utf-8

# In[6]:

import pandas as pd
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import log_loss, make_scorer
import subprocess
import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# In[2]:

train = pd.read_pickle('../data/train.pkl')
test = pd.read_pickle('../data/test.pkl')

print (train)
print (test)


# In[3]:

featureset1 = ['len_q1c', 'len_q2c', 'words_q1c', 'words_q2c', 'chars_q1c', 'chars_q2c', 'wordshare']
featureset2 = ['qratio', 'wratio', 'partial_ratio', 'partial_tokenset', 'tokenset', 'partial_tokensort']

features = featureset1 + featureset2

print (features)


# In[4]:

X_train, X_test, y_train, y_test = train_test_split(train[features], train['is_duplicate'], test_size=0.33)


# In[ ]:

accuracy = make_scorer(log_loss)

# Scaling all numeric features for SVMs
# scaler = StandardScaler()
# train[features] = scaler.fit_transform(train[features])

# svc_params = {"C": [1, 2, 3], "gamma": [0.1, 1, 2, 3, 4], "kernel": ['linear', 'rbf']}

# %time svm_grid = GridSearchCV(estimator=SVC(), param_grid=svc_params, scoring=accuracy, cv=2)
# %time svm_grid.fit(X_train[features], y_train)

# print ("SVM grid search: ")
# print ("CV results", svm_grid.cv_results_)
# print ("Best SVM", svm_grid.best_estimator_)
# print ("Best CV score for SVM", svm_grid.best_score_)
# print ("Best SVM params:", svm_grid.best_params_)

subprocess.call(['speech-dispatcher'])        #start speech dispatcher
subprocess.call(['spd-say', '"your process has finished"'])


# In[ ]:

svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

subprocess.call(['speech-dispatcher'])        #start speech dispatcher
subprocess.call(['spd-say', '"your process has finished"'])


# In[1]:

print (log_loss(y_test, y_pred))


# In[ ]:




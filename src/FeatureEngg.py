
# coding: utf-8

# In[1]:

from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pandas as pd


# In[3]:

train = pd.read_pickle('../data/train.pkl')
test = pd.read_pickle('../data/test.pkl')


# In[7]:

# Calculate fuzzy features

def calc_qratio(row):
    # using cleaned questions
    q1 = str(row[0]).lower()
    q2 = str(row[1]).lower()
    return fuzz.QRatio(q1, q2)

def calc_wratio(row):
    q1 = str(row[0]).lower()
    q2 = str(row[1]).lower()
    return fuzz.WRatio(q1, q2)


def calc_partialratio(row):
    q1 = str(row[0]).lower()
    q2 = str(row[1]).lower()
    return fuzz.partial_ratio(q1, q2)

def calc_partial_tokenset_ratio(row):
    q1 = str(row[0]).lower()
    q2 = str(row[1]).lower()
    return fuzz.partial_token_sort_ratio(q1, q2)

def calc_tokensort_ratio(row):
    q1 = str(row[0]).lower()
    q2 = str(row[1]).lower()
    return fuzz.token_sort_ratio(q1, q2)


def calc_tokenset_ratio(row):
    q1 = str(row[0]).lower()
    q2 = str(row[1]).lower()
    return fuzz.token_set_ratio(q1, q2)
    
    
def calculate_featureset2(dataframe):
    dataframe['qratio'] = dataframe.apply(calc_qratio, axis=1, raw=True)
    dataframe['wratio'] = dataframe.apply(calc_wratio, axis=1, raw=True)
    dataframe['partial_ratio'] = dataframe.apply(calc_partialratio, axis=1, raw=True)
    dataframe['partial_tokenset'] = dataframe.apply(calc_partial_tokenset_ratio, axis=1, raw=True)
    dataframe['tokenset'] = dataframe.apply(calc_tokenset_ratio, axis=1, raw=True)
    dataframe['partial_tokensort'] = dataframe.apply(calc_tokensort_ratio, axis=1, raw=True)
    return dataframe


# In[9]:

train = calculate_featureset2(train)
test = calculate_featureset2(test)

train.to_pickle('../data/train.pkl')
test.to_pickle('../data/test.pkl')


# In[ ]:




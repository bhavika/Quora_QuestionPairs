
# coding: utf-8

# In[9]:

import seaborn as sns
import pandas as pd


get_ipython().magic(u'matplotlib inline')


# In[10]:

train = pd.read_pickle('../data/train.pkl')
test = pd.read_pickle('../data/test.pkl')


# In[11]:

# The distribution of duplicate and non-duplicate question pairs
train.groupby("is_duplicate")["id"].count().plot.bar()


# In[12]:

n = 10000
sns.pairplot(train[['len_q1', 'len_q2', 'words_q1c', 'words_q2c', 'chars_q1c', 'chars_q2c', 'wordshare','is_duplicate']][0:n],
            hue='is_duplicate')


# In[ ]:




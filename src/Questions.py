
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('../data/train_f.csv', sep=';')
test = pd.read_csv('../data/test_f.csv', sep=';')

train.groupby("is_duplicate")['id'].count().plot.bar()
plt.show()
# def load_data(path):
#     train = pd.read_csv(path+'/train.csv')
#     test = pd.read_csv(path+'/test.csv')
#     y = train['is_duplicate']
#     return train, test


# train, test = load_data('../data')


# In[3]:

# print (train.head(10))
#
#
# # In[4]:
#
# q1_avg = train['len_q1c'].mean()
# q2_avg = train['len_q2c'].mean()
#
# print (q1_avg, q2_avg)
#
# # In[5]:
#
# train['len_q1c'].max()
#
#
# # In[6]:
#
# train['len_q2c'].max()
# # train.iloc[train['len_q2'].max()]
#
#
# # In[7]:
#
# not_dupes = train[train.is_duplicate == 0].count()
# dupes = train[train.is_duplicate == 1].count()
#
#
# # In[8]:
#
# print (dupes, not_dupes)
# print (dupes/(dupes+not_dupes) * 100)
#
#
# # In[11]:
#
# train.loc[train.len_q1 > q1_avg]
#
#
# # In[17]:
#
# train['len_q1c'].min()
#
#
# # In[18]:
#
# train['len_q2c'].min()
#
#
# # In[9]:
#
# question1_vectors = np.zeros((train.shape[0], 300))
#
#
# # In[12]:
#
# print (question1_vectors[0].shape)
#
#
# # In[13]:
#
# import gensim
# # wv_model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin.gz', binary=True)
#
#
# # In[15]:
#
# # print wv_model['India'].shape
#
#
# # In[39]:
#
# from nltk import word_tokenize
# from nltk.corpus import stopwords
#
# stop_words = stopwords.words('english')
#
# def sent2vec(s):
#     words = str(s).lower().decode('utf-8')
#     words = word_tokenize(words)
#     words = [w for w in words if not w in stop_words]
#     # print words
#     M = []
#     for w in words:
#         try:
#             M.append(wv_model[w])
#         except:
#             continue
#     M = np.array(M)
#     v = M.sum(axis = 0)
#     print (v)
#     return v/ np.sqrt((v**2).sum())
#
#
# x = sent2vec("India is my country elephant")
# y = sent2vec("Elephants are in my country India")
#
# # print x.shape
# # print y.shape
#
#
# # In[40]:
#
# from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
#
# # print cosine(x,y)
# # print cityblock(x,y)
# # print jaccard(x,y)
# # print canberra(x,y)
# # print euclidean(x,y)
# # print minkowski(x,y,3)
# # print braycurtis(x,y)
#

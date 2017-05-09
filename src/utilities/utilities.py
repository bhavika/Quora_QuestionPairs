import pandas as pd
from nltk.corpus import stopwords
import os
import numpy as np

stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',
              'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',
              'Is','If','While','This']

stop_words = stopwords.words('english')


DATA = '../data'
TRAIN_PKL = '../data/train_f.csv'
TEST_PKL = '../data/test_f.csv'

Q1_VECTORS_TRAIN = '../data/q1_w2v_train.npy'
Q2_VECTORS_TRAIN = '../data/q2_w2v_train.npy'

Q1_VECTORS_TEST = '../data/q1_w2v_test.npy'
Q2_VECTORS_TEST = '../data/q2_w2v_test.npy'

# LSTM Settings

seq_length = 30
max_nb_words = 200000
embedding_dim = 300
re_weight = True


def load_data(path):
    train = pd.read_csv(path+'/train.csv')
    test = pd.read_csv(path+'/test.csv')

    train = pd.DataFrame(train)
    test = pd.DataFrame(test)

    y = train['is_duplicate']
    return train, test


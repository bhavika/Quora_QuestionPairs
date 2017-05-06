import pandas as pd
import numpy as np
from gensim.models import word2vec
from tqdm import tqdm, tqdm_pandas
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
# from keras.engine.topology import Merge
from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import joblib
import gensim
from time import time


start = time()

MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25

wv_model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin.gz', binary=True)
norm_model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin.gz', binary=True)
norm_model.init_sims(replace=True)

STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
        rate_drop_dense)

train = joblib.load('../data/train.pkl')
test = joblib.load('../data/test.pkl')

q1 = train['question1'].tolist()
q2 = train['question2'].tolist()
q1_test = test['question1'].tolist()
q2_test = test['question2'].tolist()
labels = train['is_duplicate'].tolist()
ids = test['test_id']

re_weight = True #for imbalance

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(q1 + q2 + q1_test + q2_test)

sequences_1 = tokenizer.texts_to_sequences(q1)
sequences_2 = tokenizer.texts_to_sequences(q2)
test_sequences_1 = tokenizer.texts_to_sequences(q1_test)
test_sequences_2 = tokenizer.texts_to_sequences(q2_test)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))
data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(labels)

# train
w2v_q1 = joblib.load('../data/q1_w2v.pkl')
w2v_q2 = joblib.load('../data/q2_w2v.pkl')

print (w2v_q1.shape)
print (w2v_q2.shape)

print ("Elapsed time till loading word vectors", time()-start)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
ids = np.array(ids)

perm = np.random.permutation(len(data_1))
idx_train = perm[:int(len(data_1)*(1-VALIDATION_SPLIT))]
idx_val = perm[int(len(data_1)*(1-VALIDATION_SPLIT)):]

data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
labels_train = np.concatenate((labels[idx_train], labels[idx_train]))

data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

nb_words = min(MAX_NB_WORDS, len(word_index))+1

weight_val = np.ones(len(labels_val))
if re_weight:
    weight_val *= 0.472001959
    weight_val[labels_val==0] = 1.309028344

start_wv = time()

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in wv_model.vocab:
        embedding_matrix[i] = wv_model.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

joblib.dump('embedding_matrix_lstm.pkl', embedding_matrix)

print ("Elapsed time till creating embedding matrix", time()-start_wv)

print ("Elapsed time", time()-start)


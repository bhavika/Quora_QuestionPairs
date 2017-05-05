import pandas as pd
import numpy as np
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

MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

train = joblib.load('../data/train.pkl')
test = joblib.load('../data/test.pkl')

q1 = train['question1'].tolist()
q2 = train['question2'].tolist()
q1_test = test['question1'].tolist()
q2_test = test['question2'].tolist()
labels = train['is_duplicate'].tolist()


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

w2v_q1 = joblib.load('../data/q1_w2v.pkl')

print w2v_q1[0]
print w2v_q1[0].shape
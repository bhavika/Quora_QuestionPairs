import pandas as pd
import numpy as np
from gensim.models import word2vec
from tqdm import tqdm, tqdm_pandas
from keras.layers import Dense, Input, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing import sequence, text
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import joblib
import gensim
from time import time
from utilities import utilities as u
import re
from string import punctuation
from tqdm import tqdm, tqdm_pandas
from keras.layers.merge import concatenate

tqdm.pandas()
start = time()

num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25

# activation function
act = 'relu'

load_wv = time()

wv_model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin.gz', binary=True)
norm_model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin.gz', binary=True)
norm_model.init_sims(replace=True)

print ("Time to load model", time() - load_wv)

STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
        rate_drop_dense)

train = pd.read_csv(u.TRAIN_PKL, sep=';')
test = pd.read_csv(u.TEST_PKL, sep=';')

train['0'] = train['0'].progress_apply(lambda x: re.sub(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?', "", str(x)))
train['1'] = train['1'].progress_apply(lambda x: re.sub(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?', "", str(x)))

q1 = train['0'].tolist()
q2 = train['1'].tolist()

test['0'] = test['0'].progress_apply(lambda x: re.sub(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?', "", str(x)))
test['1'] = test['1'].progress_apply(lambda x: re.sub(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?', "", str(x)))

q1_test = test['0'].tolist()
q2_test = test['1'].tolist()

labels = train['is_duplicate'].tolist()
ids = test['test_id'].tolist()

tokenizer = Tokenizer(num_words=u.max_nb_words)
tokenizer.fit_on_texts(q1 + q2 + q1_test + q2_test)

sequences_1 = tokenizer.texts_to_sequences(q1)
sequences_2 = tokenizer.texts_to_sequences(q2)

test_sequences_1 = tokenizer.texts_to_sequences(q1_test)
test_sequences_2 = tokenizer.texts_to_sequences(q2_test)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))
data_1 = pad_sequences(sequences_1, maxlen=u.seq_length)
data_2 = pad_sequences(sequences_2, maxlen=u.seq_length)
labels = np.array(labels)

print ("Elapsed time till loading word vectors", time()-start)

test_data_1 = pad_sequences(test_sequences_1, maxlen=u.seq_length)
test_data_2 = pad_sequences(test_sequences_2, maxlen=u.seq_length)
ids = np.array(ids)

# If Q2 is a duplicate of Q1, Q1 is also a duplicate of Q2.
# We feed this into the LSTM by stacking the two combinations
data_1_train = np.vstack((data_1, data_2))
data_2_train = np.vstack((data_2, data_1))
labels_train = np.concatenate((labels, labels))

nb_words = min(u.max_nb_words, len(word_index))+1

weight_val = np.ones(len(labels_train))
if u.re_weight:
    weight_val *= 0.472001959
    weight_val[labels_train == 0] = 1.309028344

start_wv = time()

# Create the embedding matrix here - we do this once, after that we just load it each time - uncomment line 114
# Comment out the block from 107 to 113 to avoid creating the matrix each time.
embedding_matrix = np.zeros((nb_words, u.embedding_dim))
for word, i in word_index.items():
    if word in wv_model.vocab:
        embedding_matrix[i] = wv_model.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

np.save('../data/embedding_matrix_lstm.npy', embedding_matrix)

# embedding_matrix = np.load('../data/embedding_matrix_lstm.npy')

print ("Elapsed time till creating embedding matrix", time()-start_wv)

print("Elapsed time", time()-start)

start_model = time()

embedding_layer = Embedding(nb_words,
        u.embedding_dim,
        weights=[embedding_matrix],
        input_length=u.seq_length,
        trainable=False)
lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

sequence_1_input = Input(shape=(u.seq_length,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(u.seq_length,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = lstm_layer(embedded_sequences_2)


m = concatenate([x1, y1])
m = Dropout(rate_drop_dense)(m)
m = BatchNormalization()(m)

m = Dense(num_dense, activation=act)(m)
m = Dropout(rate_drop_dense)(m)
m = BatchNormalization()(m)

preds = Dense(1, activation='sigmoid')(m)

# Add class weights
if u.re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None


# Train the model
model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

print(STAMP)

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([data_1_train, data_2_train], labels_train,
        validation_split=0.1,
        epochs=200, batch_size=2048, shuffle=True,
        class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])


# Create the submission file

print('Writing the submission file...')

preds = model.predict([test_data_1, test_data_2], batch_size=8192, verbose=1)
preds += model.predict([test_data_2, test_data_1], batch_size=8192, verbose=1)
preds /= 2

submission = pd.DataFrame({'test_id':ids, 'is_duplicate':preds.ravel()})
submission.to_csv('%.4f_'%(bst_val_score)+STAMP+'.csv', index=False)

print ("Elapsed time", time()-start_model)
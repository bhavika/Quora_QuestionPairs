from keras.layers import Dense, Input, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import numpy as np
import utilities as u
from keras.layers.merge import concatenate

num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25
act = 'relu' # activation function

seq_length = 30
max_nb_words = 200000
embedding_dim = 300
re_weight = True

len_word_index = 83617

nb_words = min(max_nb_words, len_word_index)+1

embedding_matrix = np.load('../data/embedding_matrix_lstm.npy')


def create_model():
    embedding_layer = Embedding(nb_words,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=seq_length,
                                trainable=False)
    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

    sequence_1_input = Input(shape=(seq_length,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(seq_length,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    m = concatenate([x1, y1])
    m = Dropout(rate_drop_dense)(m)
    m = BatchNormalization()(m)

    m = Dense(num_dense, activation=act)(m)
    m = Dropout(rate_drop_dense)(m)
    m = BatchNormalization()(m)

    preds = Dense(1, activation='sigmoid')(m)

    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

    print(model.summary())


create_model()
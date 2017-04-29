
# coding: utf-8

from tqdm import tqdm
from fuzzywuzzy import fuzz
import cPickle
import pandas as pd
import gensim
import numpy as np
from nltk.corpus import stopwords
from nltk import word_tokenize

stop_words = stopwords.words('english')

train = pd.read_pickle('../data/train.pkl')
test = pd.read_pickle('../data/test.pkl')

wv_model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin.gz', binary=True)
norm_model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin.gz', binary=True)
norm_model.init_sims(replace=True)


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


def calc_wordmoversdist(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return wv_model.wmdistance(s1, s2)


def calc_norm_wordmover(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)


def sent2vec(s):
    words = str(s).lower().decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    M = []
    for w in words:
        try:
            M.append(wv_model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis = 0)
    return v/ np.sqrt((v**2).sum())


def calc_question_vectors(data):
    question1_vectors = np.zeros((data.shape[0], 300))
    question2_vectors = np.zeros((data.shape[0], 300))
    for i, q in tqdm(enumerate(data.question1.values)):
        question1_vectors[i, :] = sent2vec(q)
    for i, q in tqdm(enumerate(data.question2.values)):
        question2_vectors[i, :] = sent2vec(q)
    return question1_vectors, question2_vectors


def calculate_featureset3(dataframe):
    # Word Mover's Distance
    dataframe['wmd'] = dataframe.apply(lambda x: calc_wordmoversdist(x['question1'], x['question2']), axis =1)
    dataframe['norm_wmd'] = dataframe.apply(lambda x: calc_norm_wordmover(x['question1'], x['question2']), axis =1)
    return dataframe

train = calculate_featureset3(train)
test = calculate_featureset3(test)

q1_vector, q2_vector = calc_question_vectors(train)

cPickle.dump(q1_vector, open('../data/q1_w2v.pkl', 'wb'), -1)
cPickle.dump(q2_vector, open('../data/q2_w2v.pkl', 'wb'), -1)

train.to_pickle('../data/train.pkl')
test.to_pickle('../data/test.pkl')






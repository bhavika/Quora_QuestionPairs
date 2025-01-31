# coding: utf-8

import pandas as pd
import numpy as np
import nltk
import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import difflib
from nltk import word_tokenize
from tqdm import tqdm, tqdm_pandas
from fuzzywuzzy import fuzz
import pandas as pd
import gensim
import numpy as np
from scipy.stats import skew, kurtosis
import pyemd
from time import time
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from utilities import utilities as u

wv_model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin.gz', binary=True)
norm_model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin.gz', binary=True)
norm_model.init_sims(replace=True)

start = time()

def fill_missing_values(train, test):
    # Check for any null values
    print(train.isnull().sum())
    print(test.isnull().sum())
    
    # We find 2 null values in train and test both
    # Replace them with an 'empty' string
    train = train.fillna('empty')
    test = test.fillna('empty')
    return train, test

def clean_text(text, remove_stopwords=True, stemming=False):

     # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text) 
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)  
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text) 
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text) 
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text) 
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text) 
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)
    
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Remove stop words
    if remove_stopwords:
        text = text.split()
        text = [w for w in text if not w in u.stop_words]
        text = " ".join(text)
    
    # Shorten words to their stems
    if stemming:
        text = text.split()
        stemmer = PorterStemmer()
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)


train, test = u.load_data(u.DATA)

train, test = fill_missing_values(train, test)
print (type(train), type(test))

# print(train.isnull().sum())
# print(test.isnull().sum())


def process_questions(question_list, questions, question_list_name, dataframe):
    '''transform questions and display progress'''
    for question in tqdm(questions):
        question_list.append(clean_text(question, remove_stopwords=True, stemming=True))
        # if len(question_list) % 100000 == 0:
        #     progress = len(question_list)/len(dataframe) * 100
        #     print("{} is {}% complete.".format(question_list_name, round(progress, 1)))

train_question2 = []
process_questions(train_question2, train.question2, 'train_question2', train)

train_question1 = []
process_questions(train_question1, train.question1, 'train_question1', train)

test_question1 = []
process_questions(test_question1, test.question1, 'test_question1', test)

test_question2 = []
process_questions(test_question2, test.question2, 'test_question2', test)

#append these clean questions back to the original dataframes

train_q1_clean = pd.Series(train_question1)
train_q2_clean = pd.Series(train_question2)
test_q1_clean = pd.Series(test_question1)
test_q2_clean = pd.Series(test_question2)

train = pd.concat([train, train_q1_clean, train_q2_clean],  axis=1)
test = pd.concat([test, test_q1_clean, test_q2_clean], axis=1)

print ("Train is", type(train), "test is ", type(test))


# Start feature engineering

def calculate_wordshare(row):
    q1_words = {}
    q2_words = {}
    
    # getting words from question 1 and 2 
    for word in str(row['question1']).lower().split(" "):
        q1_words[word] = 1
    for word in str(row['question2']).lower().split(" "):
        q2_words[word] = 1
    if len(q1_words) == 0 and len(q2_words) == 0:
        return 0
    common_words_q1 = [w for w in q1_words.keys() if w in q2_words]
    common_words_q2 = [w for w in q1_words.keys() if w in q2_words]
    wordshare = 1.0 * (len(common_words_q1) + len(common_words_q2))/(len(q1_words) + len(q2_words))
    return wordshare


def calculate_featureset1(dataframe):

    """
    Input: DataFrame
    Description:
    'c' at the end of the feature name indicates 'clean' which is the computed questions 
    after clean_text() and process_questions() operations above. 
    
    We compute these features: 
    1) Length of question1 - len_q1
    2) Length of question2 - len_q2
    3) Length of question1 after cleaning - len_q1c
    4) Length of question2 after cleaning - len_q2c
    5) No of words in q1 after cleaning - words_q1c
    6) No of words in q2 after cleaning - words_q2c
    7) Characters in q1 after cleaning and excluding spaces - chars_q1c
    8) Characters in q2 after cleaning and excluding spaces - chars_q2c
    9) Average number of words shared between q1 and q2 - wordshare
    """

    dataframe['len_q1'] = dataframe.question1.map(lambda x: len(str(x)))
    dataframe['len_q2'] = dataframe.question2.map(lambda x: len(str(x)))
    dataframe['len_q1c'] = dataframe[0].map(lambda x: len(str(x)))
    dataframe['len_q2c'] = dataframe[1].map(lambda x: len(str(x)))

    dataframe['words_q1c'] = dataframe[0].map(lambda x: len(str(x).split(" ")))
    dataframe['words_q2c'] = dataframe[1].map(lambda x: len(str(x).split(" ")))

    # Counting the characters but excluding the spaces
    dataframe['chars_q1c'] = dataframe[0].map(lambda x: len(x) - x.count(' '))
    dataframe['chars_q2c'] = dataframe[1].map(lambda x: len(x) - x.count(' '))

    dataframe['wordshare'] = dataframe.apply(calculate_wordshare, axis=1, raw=True)

    return dataframe


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
    s1 = [w for w in s1 if w not in u.stop_words]
    s2 = [w for w in s2 if w not in u.stop_words]
    return wv_model.wmdistance(s1, s2)


def calc_norm_wordmover(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    s1 = [w for w in s1 if w not in u.stop_words]
    s2 = [w for w in s2 if w not in u.stop_words]
    return norm_model.wmdistance(s1, s2)


def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in u.stop_words]
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
    tqdm_pandas(tqdm())
    dataframe['wmd'] = dataframe.progress_apply(lambda x: calc_wordmoversdist(x['question1'], x['question2']), axis =1)
    dataframe['norm_wmd'] = dataframe.progress_apply(lambda x: calc_norm_wordmover(x['question1'], x['question2']), axis =1)
    return dataframe


def calculate_featureset4(dataframe, q1_vectors, q2_vectors):
    dataframe['cosine_dist'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(q1_vectors), np.nan_to_num(q2_vectors))]
    dataframe['cityblock_dist'] = [cityblock(x,y) for (x,y) in zip(np.nan_to_num(q1_vectors), np.nan_to_num(q2_vectors))]
    dataframe['jaccard_dist'] = [jaccard(x, y) for (x,y) in zip(np.nan_to_num(q1_vectors), np.nan_to_num(q2_vectors))]
    dataframe['canberra_dist'] =[canberra(x, y) for (x,y) in zip(np.nan_to_num(q1_vectors), np.nan_to_num(q2_vectors))]
    dataframe['euclidean_dist'] = [euclidean(x, y ) for (x,y) in zip(np.nan_to_num(q1_vectors), np.nan_to_num(q2_vectors))]
    dataframe['minkowski_dist'] = [minkowski(x, y, 3) for (x,y) in zip(np.nan_to_num(q1_vectors), np.nan_to_num(q2_vectors))]
    dataframe['braycurtis_dist'] = [braycurtis(x, y) for (x,y) in zip(np.nan_to_num(q1_vectors), np.nan_to_num(q2_vectors))]
    dataframe['skew_q1'] = [skew(x) for x in np.nan_to_num(q1_vectors)]
    dataframe['skew_q2'] = [skew(x) for x in np.nan_to_num(q2_vectors)]
    dataframe['kurtosis_q1'] = [kurtosis(x) for x in np.nan_to_num(q1_vectors)]
    dataframe['kurtosis_q2'] = [kurtosis(x) for x in np.nan_to_num(q2_vectors)]
    return dataframe


def scale(dataframe, features):
    print ("Before scaling")
    print (dataframe[features])
    X = MinMaxScaler().fit_transform(dataframe[features])
    print ("After scaling")
    print (dataframe[features])


print ("Making featureset 1 - Lengths and Wordshare")
train = calculate_featureset1(train)
test = calculate_featureset1(test)


print ("Train is", type(train), "test is ", type(test))


print ("Making featureset 2 - fuzzy")
train = calculate_featureset2(train)
test = calculate_featureset2(test)


print ("Train is", type(train), "test is ", type(test))

print ("Making featureset 3 - WMD")
train = calculate_featureset3(train)
test = calculate_featureset3(test)

print ("Calculating word vectors for train")
q1, q2 = calc_question_vectors(train)

print ("Saving word vectors for train")
np.save(u.Q1_VECTORS_TRAIN, q1)
np.save(u.Q2_VECTORS_TRAIN, q2)


print ("Calculating word vectors for test")
q1_test, q2_test = calc_question_vectors(test)

print ("Saving word vectors for test")
np.save(u.Q1_VECTORS_TEST, q1_test)
np.save(u.Q2_VECTORS_TEST, q2_test)

print ("Calculating featureset 4 for train")
train = calculate_featureset4(train, q1, q2)

print ("Calculating featureset 4 for test")
test = calculate_featureset4(test, q1_test, q2_test)

print ("Train is", type(train), "test is ", type(test))

print ("Dumping train and test pickles after FS-4")

train.to_csv(u.TRAIN_PKL, sep=';', header=True)
test.to_csv(u.TEST_PKL, sep=';', header=True)

print(train.columns.values)
print(test.columns.values)

print (train)
print (test)

print ("Elapsed time", time()-start)

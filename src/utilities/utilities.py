import pandas as pd
from nltk.corpus import stopwords


stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',
              'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',
              'Is','If','While','This']

stop_words = stopwords.words('english')


DATA = '../data'
TRAIN_PKL = '../data/train.pkl'
TEST_PKL = '../data/test.pkl'

Q1_VECTORS_TRAIN = '../data/q1_w2v_train.pkl'
Q2_VECTORS_TRAIN = '../data/q2_w2v_train.pkl'

Q1_VECTORS_TEST = '../data/q1_w2v_test.pkl'
Q2_VECTORS_TEST = '../data/q2_w2v_test.pkl'


def load_data(path):
    train = pd.read_csv(path+'/train.csv')
    test = pd.read_csv(path+'/test.csv')
    y = train['is_duplicate']
    return train, test

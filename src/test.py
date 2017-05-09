import numpy as np
import pandas as pd

# em = np.load('../data/embedding_matrix_lstm.npy')
#
# q1_w2v = np.load('../data/q1_w2v_train.npy')
# q2_w2v = np.load('../data/q2_w2v_train.npy')
#
# q1_w2v_test = np.load('../data/q1_w2v_test.npy')
# q2_w2v_test = np.load('../data/q2_w2v_test.npy')

# print (q1_w2v.shape)
# print (q1_w2v.size)
#
# print (q1_w2v_test.size)
# print (q1_w2v_test.shape)

# s2v_train = np.concatenate((q1_w2v, q2_w2v), axis=1)
# s2v_test = np.concatenate((q1_w2v_test, q2_w2v_test), axis=1)
#
# print (s2v_test.shape)
# print (s2v_train.shape)


train = pd.read_csv('../data/train_f.csv', sep = ';')
test = pd.read_csv('../data/test_f.csv', sep = ';')

print (train.columns.values)
print (test.columns.values)
#
# other_features_train = train[['wordshare', 'wmd', 'norm_wmd', 'cosine_dist', 'euclidean_dist']]
# other_features_test = test[['wordshare', 'wmd', 'norm_wmd', 'cosine_dist', 'euclidean_dist']]
#
# other_features_train = other_features_train.values.tolist()
# other_features_test = other_features_test.values.tolist()
#
# print (len(other_features_train))
# print (len(other_features_test))




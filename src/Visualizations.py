# coding: utf-8

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_pickle('../data/train.pkl')
test = pd.read_pickle('../data/test.pkl')

# print train.columns.values
# print train[['wmd', 'norm_wmd']]

# The distribution of duplicate and non-duplicate question pairs
# train.groupby("is_duplicate")["id"].count().plot.bar()

# dups = train[0:10000]
# print dups[dups.is_duplicate==0].count()

n = 10000


#sns.pairplot(train[['len_q1', 'len_q2', 'is_duplicate']][0:n],
#             hue='is_duplicate')


# sns.set_style("ticks")

# sns.lmplot('len_q1', 'len_q2',
#            data=train[0:10000],
#            fit_reg=False,
#            hue='is_duplicate')
#

# sns.lmplot('words_q1c', 'words_q2c',
#            data=train[0:10000],
#            fit_reg=False,
#            hue='is_duplicate')

sns.violinplot(x='is_duplicate', y='wordshare', data=train[0:2000])

plt.show()


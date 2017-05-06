import pandas as pd

sub = pd.read_csv('../data/sample_submission.csv')
results = pd.read_csv('blended3.txt')

print sub.shape
print results.shape

print sub.head(5)
print results.head(5)

submission = pd.concat([sub, results], axis = 1)
submission.columns = ['test_id', 'dup1', 'dup2', 'is_duplicate']

submission = submission.drop(['dup1', 'dup2'], 1, inplace=False)
print submission
submission.to_csv('blended3_sub.csv', index=False)
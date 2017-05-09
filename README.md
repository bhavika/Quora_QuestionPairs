### Quora Question Pairs

[Kaggle Competition] (https://www.kaggle.com/c/quora-question-pairs)

## Setup:

1) Unzip Project_btekwani.zip
2) cd/Project_btekwani
3) If you want to create a virtual environment, run virtualenv <the name of the environment, say 'tekwani'>
   Creating a virtualenv within Anaconda is highly recommended because of tensorflow, keras and their dependencies.
   To begin using the virtual environment, you must activate it. $ source tekwani/bin/activate
4)Now install the packages specified in requirements.txt. You can do this using pip freeze > requirements.txt (freeze the current state of the environment)
    `pip install -r requirements.txt`
5) Depending on your installation, NLTK might require the Porter Stemmer and stopwords data. To install these, run python.
`sudo python -m nltk.downloader -d /usr/local/share/nltk_data all`


## Files:

1) The `src` folder contains all the files used to generate visualizations, statistics, models and features used throughout the report.
2) `blend.py` is the blended model made of 6 regressors.
3) `features.py` does all the feature engineering - FS-1 through FS-4 and generates 4 .npy files which correspond to word embedding
vectors in train (q1 and q2) and test (q1 and q2). It also creates train_f.csv and test_f.csv which contain the cleaned text and
all the computed features.
4) `utilities` is a module containing paths used for saving and loading Numpy arrays, csv files, loading data as pandas DataFrames and
LSTM settings (constant values).
5) `makesub.py` is a hack to create a submission file after `blend.py` has finished executing.
6) `Questions.py` has simple pandas operations used to do basic counts, calculate average question lengths etc.
7) `XGB_Baseline.py` is the XGBoost model that is the baseline submission I made.
8) `XGBoost_GridSearch.py` - a custom wrapper around xgboost so that we can do a grid search like scikit-learn allows.
9) `Visualizations.py` - generates violin plots and bar plots used in the report. Probably will not run as it is.


## Directory structure:

Project_btekwani:
    - data
        - train.csv
        - sample_submission.csv
        - test.csv
    - report
    - src
        -utilities
            - __init__.py
            - utilities.py
        - other *.py files
    - sub
        [generated submission files go here]
    - viz
        [plots and stuff]
    GoogleNews-vectors-negative300.bin.gz


Output:

1) `xgb_gridsearch.out` - contains the GridSearch logs
2) `sub/XGB_Baseline_logs.txt` - contains XGB outputs for various features
3) `lstm_274_118_0.20_0.37.h5` - the weights for the LSTM, will require an hd5 library in Python to open and view.
4) All submitted files are in the `sub` folder.
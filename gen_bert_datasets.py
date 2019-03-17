import random
import os
import numpy as np
import pandas as pd
import sklearn.model_selection

import string

from TweetDataset import TweetDataset
from vocab import VocabEntry
import string

UNK = '<unk>'
PAD = '<pad>'

TWEET_LEN = 20
EMBED_LEN = 100
N_EPOCHS = 100

PATH = './data/training.1600000.processed.noemoticon.csv'

def load_datasets(path):
    print ("Loading dataset...")
    full_data = pd.read_csv(path, encoding = 'latin-1', names = ["Pos_Neg", "ID", "Date", "QUERY", "User", "Content"])
    full_n = full_data.shape[0]
    #print (train_data)

    train_data, dev_data = sklearn.model_selection.train_test_split(full_data, test_size = 0.04)

    # Get ground truth x and y values
    train_x_raw = train_data.loc[:]["Content"]
    train_y = [0.0 if y == 0 else 1.0 for y in train_data.loc[:]["Pos_Neg"]]

    dev_x_raw = dev_data.loc[:]["Content"]
    dev_y = [0.0 if y == 0 else 1.0 for y in dev_data.loc[:]["Pos_Neg"]]
    #print (dev_y)

    df_train = pd.DataFrame(data = {'col1' : [i + 1 for i in range(len(train_y))], 'col2' : train_y, 'col3' : ['a' for i in range(len(train_y))], 'col4' : train_x_raw})
    df_train.to_csv('./train.tsv', sep='\t', index=False, header=False)

    df_dev = pd.DataFrame(data = {'col1' : [i + 1 for i in range(len(dev_y))], 'col2' : dev_y, 'col3' : ['a' for i in range(len(dev_y))], 'col4' : dev_x_raw})
    df_dev.to_csv('./dev.tsv', sep='\t', index=False, header=False)

load_datasets(PATH)

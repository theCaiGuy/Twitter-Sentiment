import numpy as np
import pandas as pd
import random
import sklearn

# from sklearn.cross_validation import train_test_split - deprecated
from sklearn.model_selection import train_test_split

full_data_path = './data/training.1600000.processed.noemoticon.csv'

total_size = sum(1 for line in open(full_data_path, encoding = 'latin-1')) - 1

train_n = 320
train_skip = [x for x in range(1, total_size) if x % train_n != 0]
train_data = pd.read_csv(full_data_path, skiprows=train_skip, encoding = 'latin-1', names = ["Pos_Neg", "ID", "Date", "QUERY", "User", "Content"])
train_data.to_csv('train_mini.csv', encoding='latin-1')

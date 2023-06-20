#import libraries

'''%matplotlib inline'''
import logging
import time
from platform import python_version
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable

#load and read the file

df = pd.read_csv('data/train.csv')

#make experiment repeatable and shuffle the dataset
#for reducing varience to ensure that the model won’t overfit to the sequence of samples in the training set

np.random.seed(42)

df = df.sample(frac=1)
df = df.reset_index(drop=True)
df.head()
df.comment_text[0]#load the first comment

#set the limit of trainset to 10000 comments for training neural network

df_train = df[:10000].reset_index(drop=True)
df_val = df[10000:11000].reset_index(drop=True)
df_test = df[11000:13000].reset_index(drop=True)

#calling B.E.R.T
from bert import *
bertfncall()
#calling trainer
from trainer import *
trainerfncall()

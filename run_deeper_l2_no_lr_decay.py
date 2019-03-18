import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import random
import os
import pandas as pd
import numpy as np

import string
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

from TweetDataset import TweetDataset
from vocab import VocabEntry

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import sklearn
from sklearn.metrics import f1_score

from TweetDataset import TweetDataset
from vocab import VocabEntry
from convblockdeeper import ConvBlock
from DeeperCNN import DeepCNN

import torch.optim as optim

# SEED = 1234
UNK = '<unk>'
PAD = '<pad>'

TWEET_LEN = 20
EMBED_LEN = 100
N_EPOCHS = 100

# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True


PATH = './data/training.1600000.processed.noemoticon.csv'
# PATH = './data/train_mini.csv'
MODEL_PATH = './model_cache/cache_deeper_l2_no_lr_decay'

def vectorize(examples, tok2id):
    vec_examples = []
    for ex in examples:
        #print (ex)
        sentence = []
        for w in word_tokenize(ex):
            if w in string.punctuation:
                continue
            if w in tok2id:
                sentence.append(tok2id[w])
        if len(sentence) < TWEET_LEN:
            sentence += [tok2id[PAD] for i in range(TWEET_LEN - len(sentence))]
        else:
            sentence = sentence[:TWEET_LEN]
        vec_examples.append(sentence)
    return vec_examples



def build_embeddings(tok2id, char_vectors):
    embeddings_matrix = np.asarray(np.random.normal(0, 0.9, (len(tok2id), EMBED_LEN)), dtype='float32')

    for token in tok2id:
        i = tok2id[token]
        if token in char_vectors:
            embeddings_matrix[i] = char_vectors[token]
        elif token.lower() in char_vectors:
            embeddings_matrix[i] = char_vectors[token.lower()]

    return embeddings_matrix



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

    print ("Loading character vectors...")
    char_vectors = {}
    for line in open('./glove.twitter.27B/glove.twitter.27B.100d.txt').readlines():
        sp = line.strip().split()
        if len(sp) == 0: continue
        char_vectors[sp[0]] = [float(x) for x in sp[1:]]

    tok2id = {}
    for ex in full_data['Content']:
        for w in word_tokenize(ex):
            if w in string.punctuation:
                continue
            if not w in tok2id:
                tok2id[w] = len(tok2id)

    tok2id[UNK] = len(tok2id)
    tok2id[PAD] = len(tok2id)

    print ("Generating dataset objects...")
    train_x = vectorize(train_x_raw, tok2id)

    dev_x = vectorize(dev_x_raw, tok2id)

    train_dataset = TweetDataset(train_x, train_y)
    dev_dataset = TweetDataset(dev_x, dev_y)

    embeddings = build_embeddings(tok2id, char_vectors)

    return train_dataset, dev_dataset, embeddings



def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum()/len(correct)
    return acc


def F1(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds)).detach()
    return f1_score(y, rounded_preds)


def train(model, train_loader, optimizer, criterion, iter):

    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0

    model.train()

    for batchnum, batch in enumerate(train_loader):
        #print ("Training on batch #" + str(batchnum))
        train_x = torch.stack(batch['content']).cuda()
        #print (train_x.shape)
        train_y = batch['label'].float().cuda()
        #train_y = batch['label'].long()
        if train_x.shape[1] == 1: continue
        #print (train_y.view(-1).shape)
        predictions = model.forward(train_x).squeeze(1)
        #print (predictions.shape)
        loss = criterion(predictions, train_y)
        # print (loss)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        acc = binary_accuracy(predictions, train_y)
        #optimizer.step()
        f1 = F1(predictions, train_y)

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_f1 += f1.item()

        print ("| Iteration: " + str(iter + 1) + " | Train Loss: " + str(loss.item()) + " | Train Acc: " + str(acc.item()) + " | Train F1: " + str(f1.item()) + " |")
        file = open("./train_loss_deeper_l2_no_lr_decay.txt", "a")
        file.write(" iter: " + str(iter + 1) + " loss: " + str(loss.item())  + " accuracy: " + str(acc.item()) + " f1: " + str(f1.item())+ '\n')
        iter = iter + 1

    return epoch_loss / len(train_loader), epoch_acc / len(train_loader), epoch_f1 / len(train_loader), iter



def evaluate(model, dev_loader, criterion):

    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0

    model.eval()

    with torch.no_grad():

        for batchnum, batch in enumerate(dev_loader):
            dev_x = torch.stack(batch['content']).cuda()
            #print (train_x)
            dev_y = batch['label'].float().cuda()
            predictions = model(dev_x).squeeze(1)
            #print (torch.round(predictions))
            loss = criterion(predictions, dev_y)
            acc = binary_accuracy(predictions, dev_y)
            f1 = F1(predictions, dev_y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_f1 += f1.item()

    return epoch_loss / len(dev_loader), epoch_acc / len(dev_loader), epoch_f1 / len(dev_loader)


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_model(train_dataset, dev_dataset, embeddings_matrix):
    lr = 0.0002
    CNN_model = DeepCNN(embeddings_matrix)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.Adam(CNN_model.parameters(), lr = lr, weight_decay = 0.001)
    criterion = nn.BCEWithLogitsLoss()
    CNN_model = CNN_model.to(device)
    criterion = criterion.to(device)

    train_loader = DataLoader(train_dataset,
                          batch_size=2048,
                          shuffle=True,
                          num_workers=4
                         )

    dev_loader = DataLoader(dev_dataset,
                      batch_size=2048,
                      shuffle=False,
                      num_workers=4
                     )

    best_valid_f1 = 0.0

    print ("Training model...")

    iter = 0

    torch.save(CNN_model.state_dict(), MODEL_PATH)
    torch.save(optimizer.state_dict(), MODEL_PATH + '.optim')

    for epoch in range(N_EPOCHS):

        train_loss, train_acc, train_f1, iter = train(CNN_model, train_loader, optimizer, criterion, iter)

        print ('Epoch ' + str(epoch + 1) + " completed. Average minibatch train loss: " + str(train_loss) + ". Average minibatch train acc: " + str(train_acc) + ". Average minibatch F1: " + str(train_f1))

        print ("Begin validation...")
        valid_loss, valid_acc, valid_f1 = evaluate(CNN_model, dev_loader, criterion)
        print ("| Val. Loss: " + str(valid_loss) + " | Val. Acc: " + str(valid_acc) + " | Val. F1: " + str(valid_f1) + " |")
        file = open("./val_loss_deeper_l2_no_lr_decay.txt", "a")
        file.write("epoch: " + str(epoch + 1) + " iter: " + str(iter + 1) + " loss: " + str(valid_loss) + " accuracy: " + str(valid_acc) + " f1: " + str(valid_f1) + '\n')

        if valid_f1 > best_valid_f1:
            print ("New top validation f1 score achieved! Saving model params...")
            torch.save(CNN_model.state_dict(), MODEL_PATH)
            torch.save(optimizer.state_dict(), MODEL_PATH + '.optim')
            best_valid_f1 = valid_f1

    print ("Training complete! Best validation f1 score = " + str(best_valid_f1))

    return CNN_model

if __name__ == '__main__':
    train_dataset, dev_dataset, embeddings_matrix = load_datasets(PATH)
    CNN_model = train_model(train_dataset, dev_dataset, embeddings_matrix)

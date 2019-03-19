import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from convblockdeeper import ConvBlock

class DeepCNN(nn.Module):
    def __init__(self, embeddings, dropout=0.5):
        super().__init__()

        self.batch_size = embeddings.shape[1]

        # Freeze pretrained GloVe embeddings
        self.embedding = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(embeddings))
        self.embedding.weight.requires_grad = False

        self.conv = nn.Conv1d(in_channels=embeddings.shape[1], out_channels=128, kernel_size=5, padding = 2)
        #torch.nn.init.xavier_uniform_(self.conv.weight)
        self.block = ConvBlock(128, 256)
        self.block2 = ConvBlock(256, 256)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(5 * 256, 256)
        #self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 1)
        #torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x, print_sizes = False):
        # x.shape = (sent_len, batch_size)
        x = x.permute(1, 0)
        if print_sizes: print ("x.shape: " + str(x.shape))
        # x.shape = (batch_size, sent_len)

        x_emb = self.embedding(x).permute(0, 2, 1)
        if print_sizes: print ("x_emb.shape: " + str(x_emb.shape))
        # x_emb.shape = (batch_size, emb_size, sent_len)

        x_conv = self.conv(x_emb)
        # x_conv.shape = (batch_size, 128, sent_len)
        if print_sizes: print ("x_conv.shape: " + str(x_conv.shape))

        x_block = self.block.forward(x_conv)
        # x_block.shape = (batch_size, 128, sent_len / 2)
        if print_sizes: print ("x_block.shape: " + str(x_block.shape))

        x_block_2 = self.block2.forward(x_block)
        # x_block.shape = (batch_size, 256, sent_len / 4)
        if print_sizes: print ("x_block_2.shape: " + str(x_block_2.shape))

        x_cat = x_block_2.view(-1, x_block_2.shape[1] * x_block_2.shape[2])
        # x_cat.shape = (batch_size, 128 * sent_len / 2)
        if print_sizes: print ("x_cat.shape: " + str(x_cat.shape))

        x_fc = self.fc(self.dropout(x_cat))
        # x_fc.shape = (batch_size, 1)
        if print_sizes: print ("x_fc.shape: " + str(x_fc.shape))

        x_fc2 = self.fc2(self.dropout2(x_fc))

        return x_fc2

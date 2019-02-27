import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, num_filters, k = 5, stride = 1):
        """ Initialize a CNN network with a kernel of size k,  """
        super().__init__()

        self.in_channels = in_channels
        self.num_filters = num_filters
        self.k = k
        self.stride = stride

        self.conv1 = nn.Conv1d(in_channels = in_channels, out_channels = num_filters, kernel_size = k, bias = True, padding = 2)
        self.batch_norm1 = nn.BatchNorm1d(num_filters)
        self.relu1 = nn.ReLU()
        #torch.nn.init.xavier_uniform_(self.conv1.weight)

        self.conv2 = nn.Conv1d(in_channels = num_filters, out_channels = num_filters, kernel_size = k, bias = True, padding = 2)
        self.batch_norm2 = nn.BatchNorm1d(num_filters)
        self.relu2 = nn.ReLU()
        #torch.nn.init.xavier_uniform_(self.conv2.weight)

        self.pool = nn.MaxPool1d(kernel_size = 2)

    def forward(self, x, print_sizes = False) -> torch.Tensor:
        """ Input x of size (batch size, embed_size, sent_len) and output of size [batchsize, embed_size] """

        x_conv1 = self.conv1(x)
        x_relu1 = self.relu1(x_conv1)
        x_bn = self.batch_norm1(x_relu1)
        if print_sizes: print (x_bn.shape)

        x_conv2 = self.batch_norm2(self.relu2(self.conv2(x_bn)))
        if print_sizes: print (x_conv2.shape)

        x_pool = self.pool(x_conv2)
        # print (x_pool)
        if print_sizes: print(x_pool.shape)

        return x_pool

# def test_all():
#     batch_size = 3
#     embed_size = 5
#     sent_len = 6
#     num_filters = 6
#
#     fake_input = torch.tensor(np.zeros((batch_size, embed_size, sent_len)), dtype = torch.float32)
#     block = ConvBlock(embed_size, num_filters)
#     print (block.forward(fake_input, True))
#
#
# if __name__ == '__main__':
#     #test_intermediate_sizes()
#     #test_output()
#     test_all()

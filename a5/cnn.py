#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

KERNEL_SIZE = 5

class CNN(nn.Module):
    def __init__(self, e_char, e_word, m_word, kernel_size=5):
        """
        performs 1d convolution to the input
        :param e_char: character embedding size, used as input channel size.
        :param e_word: word embedding size, used as output channel size.
        :param m_word: max word lenght, used to determine dimension after the convolution
        :param kernel_size: window size.
        """
        super(CNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=e_char, out_channels=e_word, kernel_size=kernel_size)
        self.maxpool = nn.MaxPool1d(kernel_size=m_word-kernel_size+1)

    def forward(self, x_reshape: torch.tensor) -> torch.tensor:
        """
        :param x_reshape: a tensor of shape (max_sentence_len * batch_size, char_embedding_size, max_word_len)
        :return: a tensor of shape (max_sentence_len * batch_size, word_embedding_size)
        """
        x_conv = self.conv1d(x_reshape)  # (max_sentence_len * batch_size, word_embedding_size, max_word_len - kernel_size + 1)
        x_conv_relu = torch.relu(x_conv)  # same as above
        x_conv_out = self.maxpool(x_conv_relu)  # (max_sentence_len * batch_size, word_embedding_size, 1)
        return torch.squeeze(x_conv_out)  # (max_sentence_len * batch_size, word_embedding_size)

### END YOUR CODE


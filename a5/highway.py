#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self, embedding_size):
        """
        performs gate operation on the input
        :param embedding_size: used as the dimension of the weights.
        """
        super(Highway, self).__init__()

        self.e_projection = nn.Linear(embedding_size, embedding_size, bias=True)
        self.e_gate = nn.Linear(embedding_size, embedding_size, bias=True)

    def forward(self, conv_out: torch.Tensor) -> torch.Tensor:
        """
        :param conv_out: tensor of integers of shape (max_sentence_len, batch_size, word_embedding_size)
        :return: tensor of shape (max_sentence_len, batch_size, word_embedding_size)
        """
        x_proj = self.e_projection(conv_out)
        x_proj_relu = torch.relu(x_proj)
        x_gate = self.e_gate(conv_out)
        x_gate_sigmoid = torch.sigmoid(x_gate)
        x_highway = x_gate_sigmoid * x_proj_relu + (1-x_gate_sigmoid) * conv_out
        return x_highway

### END YOUR CODE 


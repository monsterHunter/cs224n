#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        pad_token_idx = vocab.char2id['<pad>']
        self.char_embed_size = 50
        self.dropout_rate = 0.3
        self.max_word_size = 21
        self.word_embed_size = embed_size
        self.embed_size = embed_size
        self.v_char = len(vocab.char2id)
        self.v_word = len(vocab.word2id)

        self.embeddings = nn.Embedding(self.v_char, self.char_embed_size, padding_idx=pad_token_idx)
        self.Dropout = nn.Dropout(p=self.dropout_rate)
        self.cnn = CNN(e_char=self.char_embed_size, e_word=self.word_embed_size, m_word=self.max_word_size)
        self.highway = Highway(embedding_size=self.word_embed_size)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        max_word_size = input.size()[-1]
        assert(max_word_size == self.max_word_size)

        char_embeddings = self.embeddings(input)  # (max_sent_len, batch_size, max_word_len, char_embedding_size)
        # conv1d only performs on the last dimension so we have to swap
        char_embeddings = char_embeddings.permute(0, 1, 3, 2)  # (max_sent_len, batch_size, char_embedding_size, max_word_len)

        max_sent_len = char_embeddings.size()[0]
        batch_size = char_embeddings.size()[1]
        char_embedding_size = char_embeddings.size()[2]
        max_word_len = char_embeddings.size()[3]
        # conv1d only accepts 3 dimension array, so any extra dimensions need to be concatenated.
        char_embeddings = char_embeddings.reshape(max_sent_len * batch_size, char_embedding_size, max_word_len)  # (max_sent_len * batch_size, char_embedding_size, max_word_len)
        cnn_out = self.cnn.forward(char_embeddings)  # (max_sent_len * batch_size, word_embedding_size)

        highway_out = self.highway.forward(cnn_out)  # (max_sent_len * batch_size, word_embedding_size)
        dropout_out = self.Dropout(highway_out)
        output = dropout_out.reshape(max_sent_len, batch_size, dropout_out.size()[-1])  # (max_sent_len, batch_size, word_embedding_size)

        return output
        ### END YOUR CODE


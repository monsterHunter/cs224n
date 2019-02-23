#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()

        v_char = len(target_vocab.char2id)
        self.target_vocab = target_vocab

        self.charDecoder = nn.LSTM(input_size=char_embedding_size, hidden_size=hidden_size, bias=True, bidirectional=False)
        self.char_output_projection = nn.Linear(hidden_size, v_char, bias=True)

        self.padding_idx = target_vocab.char2id['<pad>']
        assert(self.padding_idx == 0)
        self.decoderCharEmb = nn.Embedding(v_char, char_embedding_size, padding_idx=self.padding_idx)

        ### END YOUR CODE



    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.

        char_embeddings = self.decoderCharEmb(input)
        all_hidden_state, dec_hidden = self.charDecoder(char_embeddings, dec_hidden)  # returns all hidden states, and last hidden state
        S = self.char_output_projection(all_hidden_state)

        return S, dec_hidden

        ### END YOUR CODE


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        cross_entropy = nn.CrossEntropyLoss(ignore_index=self.padding_idx, reduction='sum')
        target_seq = char_sequence[1:]
        input_seq = char_sequence[:-1]

        S, _ = self.forward(input_seq, dec_hidden)

        S = S.permute(0, 2, 1)

        return cross_entropy(S, target_seq)
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        hidden_states = initialStates
        batch_size = initialStates[0].size()[1]

        current_char = torch.tensor([[self.target_vocab.start_of_word]] * batch_size, device=device).permute(1, 0)
        output_word = None

        for i in range(max_length):
            s, hidden_states = self.forward(current_char, hidden_states)
            p = nn.functional.softmax(s, dim=-1)
            current_char = torch.argmax(p, dim=-1)
            if output_word is None:
                output_word = current_char
            else:
                output_word = torch.cat((output_word, current_char), dim=0)

        decodeWords = output_word.permute(1, 0).tolist()
        decodeWords = [[self.target_vocab.id2char[char_id] for char_id in (char_ids[:char_ids.index(self.target_vocab.end_of_word)] if self.target_vocab.end_of_word in char_ids else char_ids)] for char_ids in decodeWords]
        decodeWords = [''.join(chars) for chars in decodeWords]

        # for o in output_word:

        return decodeWords

        ### END YOUR CODE


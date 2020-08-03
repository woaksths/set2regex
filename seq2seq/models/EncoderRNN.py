import torch.nn as nn
import torch
import seq2seq
from seq2seq.util.string_preprocess import preprocessing, get_set_lengths, get_mask, get_mask2
from torch.autograd import Variable
from .baseRNN import BaseRNN

class EncoderRNN(BaseRNN):
    r"""
    Applies a multi-layer RNN to an input sequence.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encodr (defulat False)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        variable_lengths (bool, optional): if use variable length RNN (default: False)
        embedding (torch.Tensor, optional): Pre-trained embedding.  The size of the tensor has to match
            the size of the embedding parameter: (vocab_size, hidden_size).  The embedding layer would be initialized
            with the tensor if provided (default: None).
        update_embedding (bool, optional): If the embedding should be updated during training (default: False).

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)

    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`
        
    """

    def __init__(self, vocab_size, max_len, hidden_size,
                 input_dropout_p=0, dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='LSTM', variable_lengths=False,
                 embedding=None, update_embedding=True, vocab = None):
        super(EncoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = update_embedding
        self.vocab = vocab
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.input_dropout_p = input_dropout_p
        self.n_layers= n_layers
        self.rnn1 = self.rnn_cell(hidden_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
        
        self.rnn2 = self.rnn_cell(hidden_size*2 if self.bidirectional else hidden_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
        self.linear = nn.Linear(2*6*self.hidden_size if self.bidirectional else 6*self.hidden_size,\
                                2*self.hidden_size if self.bidirectional else self.hidden_size)

    
    def concat_pooling(self, hiddens):
        '''
        @param hiddens -> batch, seq_len, hidden
        '''
        seq_len = hiddens.size(1)
        avg_pool = torch.sum(hiddens, dim=1) / seq_len
        max_pool = torch.cat([torch.max(i[:], dim=0)[0].view(1,-1) for i in hiddens], dim=0)
        last_hidden = torch.stack([hidden[-1] for hidden in hiddens],dim=0)        
        return torch.cat([last_hidden, avg_pool, max_pool], dim=-1)
    
        
    def forward(self, input_var, input_lengths=None):        
        pos_input  = input_var[0] 
        pos_lengths = input_lengths[0] 
        batch_size = pos_input.size(0)
        set_size = pos_input.size(1)
        seq_len = pos_input.size(2)
        
        pos_embedded = self.embedding(pos_input)
        pos_embedded = pos_embedded.view(batch_size*set_size,seq_len, -1)
        pos_lengths = pos_lengths.reshape(-1)
        pos_input_mask = get_mask(pos_input)
 
        if self.variable_lengths:
            pos_embedded = nn.utils.rnn.pack_padded_sequence(pos_embedded, pos_lengths.cpu(), batch_first=True, enforce_sorted=False)
        pos_output, pos_hidden = self.rnn1(pos_embedded)
        if self.variable_lengths:
            pos_output, _ = nn.utils.rnn.pad_packed_sequence(pos_output, batch_first=True)

        last_hidden = pos_hidden[0].reshape(-1, batch_size, set_size, self.hidden_size)
        last_cell = pos_hidden[1].reshape(-1, batch_size, set_size, self.hidden_size)
        last_hidden = torch.mean(last_hidden, dim = 2)
        last_cell = torch.mean(last_cell, dim=2)
        hiddens = (last_hidden, last_cell)
        pos_output = pos_output.view(batch_size, set_size, pos_output.size(1), -1)
        return pos_output, hiddens, pos_input_mask

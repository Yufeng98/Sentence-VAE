import torch
import numpy as np
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import to_var


class LSTM_VAE(nn.Module):

    def __init__(self, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
                 num_layers=1, bidirectional=False):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding_size = embedding_size
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        elif rnn_type == 'lstm':
            rnn = nn.LSTM
        else:
            raise ValueError()

        self.hidden_factor = 2 if bidirectional else 1
        self.encoder_rnn = rnn(self.embedding_size, hidden_size, num_layers=num_layers,
                               bidirectional=self.bidirectional, batch_first=True)
        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.latent2hidden= nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.decoder_rnn = rnn(hidden_size * self.hidden_factor, hidden_size, num_layers=num_layers,
                               bidirectional=self.bidirectional, batch_first=True)
        self.hidden2embedding = nn.Linear(hidden_size * self.hidden_factor, self.embedding_size)

    def forward(self, input_embedding, length):
        batch_size = input_embedding.size(0)     # size on 0-dimension in input_sequence

        # encode
        # size of input_embedding is [20, 9, 32] for UM
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, length, batch_first=True)
        output_embedding, hidden = self.encoder_rnn(packed_input)
        padded_output_embedding = rnn_utils.pad_packed_sequence(output_embedding, batch_first=True)[0]
        padded_output_embedding = padded_output_embedding.contiguous()
        # if self.bidirectional or self.num_layers > 1:
        #     # flatten hidden state
        #     hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        # else:
        #     hidden = hidden.squeeze()

        # hidden -> latent space
        mean = self.hidden2mean(padded_output_embedding)
        logvar = self.hidden2logv(padded_output_embedding)
        std = torch.exp(0.5 * logvar)
        z = to_var(torch.randn([batch_size, length[0], self.latent_size]))
        z = z * std + mean

        # latent space -> hidden
        output_embedding = self.latent2hidden(z)

        # if self.bidirectional or self.num_layers > 1:
        #     # unflatten hidden state
        #     hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        # else:
        #     hidden = hidden.unsqueeze(0)

        # decode
        input_embedding = self.embedding_dropout(output_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, length, batch_first=True)
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        output_embedding = self.hidden2embedding(padded_outputs)

        return output_embedding, mean, logvar, z
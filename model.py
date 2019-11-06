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

        self.hidden_factor = (2 if bidirectional else 1) * num_layers
        self.encoder_rnn = rnn(self.embedding_size, hidden_size, num_layers=num_layers,
                               bidirectional=self.bidirectional, batch_first=True)
        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.decoder_rnn = rnn(self.embedding_size, hidden_size, num_layers=num_layers,
                               bidirectional=self.bidirectional, batch_first=True)
        self.hidden2embedding = nn.Linear(hidden_size * self.hidden_factor, self.embedding_size)

    def forward(self, input_embedding, length):
        batch_size = input_embedding.size(0)     # size on 0-dimension in input_sequence

        # encode
        # size of input_embedding is [20, 9, 32] for UM
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, length, batch_first=True)
        _, hidden = self.encoder_rnn(packed_input)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # hidden -> latent space
        mean = self.hidden2mean(hidden)
        logvar = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logvar)
        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean

        # latent space -> hidden
        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # decode
        input_embedding = self.embedding_dropout(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, length, batch_first=True)
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        output_embedding = self.hidden2embedding(padded_outputs)

        return output_embedding, mean, logvar, z

    # def inference(self, n=4, z=None):
    #
    #     if z is None:
    #         batch_size = n
    #         z = to_var(torch.randn([batch_size, self.latent_size]))
    #     else:
    #         batch_size = z.size(0)
    #
    #     hidden = self.latent2hidden(z)
    #
    #     if self.bidirectional or self.num_layers > 1:
    #         # unflatten hidden state
    #         hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
    #
    #     hidden = hidden.unsqueeze(0)
    #
    #     # required for dynamic stopping of sentence generation
    #     sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch
    #     sequence_running = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch which are still generating
    #     sequence_mask = torch.ones(batch_size, out=self.tensor()).byte()
    #
    #     running_seqs = torch.arange(0, batch_size, out=self.tensor()).long() # idx of still generating sequences with respect to current loop
    #
    #     generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()
    #
    #     t=0
    #     while(t<self.max_sequence_length and len(running_seqs)>0):
    #
    #         if t == 0:
    #             input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())
    #
    #         input_sequence = input_sequence.unsqueeze(1)
    #         input_embedding = self.embedding(input_sequence)
    #
    #         output, hidden = self.decoder_rnn(input_embedding, hidden)
    #
    #         logits = self.outputs2vocab(output)
    #         input_sequence = self._sample(logits)
    #
    #         # save next input
    #         generations = self._save_sample(generations, input_sequence, sequence_running, t)
    #
    #         # update gloabl running sequence
    #         sequence_mask[sequence_running] = (input_sequence != self.eos_idx).data
    #         sequence_running = sequence_idx.masked_select(sequence_mask)
    #
    #         # update local running sequences
    #         running_mask = (input_sequence != self.eos_idx).data
    #         running_seqs = running_seqs.masked_select(running_mask)
    #
    #         # prune input and hidden state according to local update
    #         if len(running_seqs) > 0:
    #             input_sequence = input_sequence[running_seqs]
    #             hidden = hidden[:, running_seqs]
    #
    #             running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()
    #
    #         t += 1
    #
    #     return generations, z
    #
    # def _sample(self, dist, mode='greedy'):
    #
    #     if mode == 'greedy':
    #         _, sample = torch.topk(dist, 1, dim=-1)
    #     sample = sample.squeeze()
    #
    #     return sample
    #
    # def _save_sample(self, save_to, sample, running_seqs, t):
    #     # select only still running
    #     running_latest = save_to[running_seqs]
    #     # update token at position t
    #     running_latest[:,t] = sample.data
    #     # save back
    #     save_to[running_seqs] = running_latest
    #
    #     return save_to

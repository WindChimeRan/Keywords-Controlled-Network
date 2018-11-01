import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable



class Encoder_GRU(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, max_len=35):
        super(Encoder_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=False, bidirectional=True)
        self.hidden = self.init_hidden()
        self.linear = nn.Linear(hidden_size * 2, max_len)

    def init_hidden(self, batch_size=10):
        # (num_layers * num_directions, batch, hidden_size)
        return Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidden_size))

    def forward(self, x, hidden, embedding_layer):
        # Embed word ids to vectors
        x_embedded = embedding_layer(x)
        B, T, embedding_size = x_embedded.size()
        x_embedded = x_embedded.permute(1, 0, 2)
        # encoder
        # hiddens = []
        # _, hidden = self.gru(x_embedded, hidden)

        # for i in range(T):
        #     _, hidden = self.gru(x_embedded[i].view(1, B, embedding_size), hidden)
        #     cat_bi_hidden = torch.cat((hidden[0], hidden[1]), 1).unsqueeze(1)
        #     hiddens.append(cat_bi_hidden)
        #     # hiddens.append(hidden[0].unsqueeze(1))
        #
        # hiddens = torch.cat(hiddens, 1)

        hiddens, hidden = self.gru(x_embedded.view(T, B, embedding_size), hidden)
        hiddens = hiddens.permute(1, 0, 2)
        cat_bi_hidden = torch.cat((hidden[0], hidden[1]), 1).unsqueeze(1)

        out = F.softmax(self.linear(cat_bi_hidden), dim=2)
        return out, cat_bi_hidden.permute(1, 0, 2), hiddens


class Summary_encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, max_len=35):
        super(Summary_encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=False)
        self.hidden = self.init_hidden()
        self.linear = nn.Linear(hidden_size, max_len)

    def init_hidden(self, batch_size=10):
        # (num_layers * num_directions, batch, hidden_size)
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))

    def forward(self, x, hidden, embedding_layer):
        # Embed word ids to vectors
        x_embedded = embedding_layer(x)
        B, T, embedding_size = x_embedded.size()
        x_embedded = x_embedded.permute(1, 0, 2)
        # encoder
        hiddens = []
        # _, hidden = self.gru(x_embedded, hidden)

        for i in range(T):
            out, hidden = self.gru(x_embedded[i].view(1, B, embedding_size), hidden)
            hiddens.append(hidden[0].unsqueeze(1))
        #
        hiddens = torch.cat(hiddens, 1)

        # out = F.log_softmax(self.linear(hidden), dim=2)
        # h = hiddens.gather(1, x_lengths).permute(1, 0, 2)
        return out, hidden, hiddens


class Summary_decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size):
        super(Summary_decoder, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(embed_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, embedding_layer):

        B = input.size(0)
        output = embedding_layer(input).view(1, B, -1)
        output = F.leaky_relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        return result
        # if use_cuda:
        #     return result.cuda()
        # else:
        #     return result

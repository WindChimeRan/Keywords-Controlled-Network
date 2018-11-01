# https://github.com/mjc92/CopyNet/blob/master/models/copynet.py
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import time


class CopyEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(CopyEncoder, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)

        self.gru = nn.GRU(input_size=embed_size,
                          hidden_size=hidden_size, batch_first=True,
                          bidirectional=True)

    def forward(self, x):
        # input: [b x seq]
        embedded = self.embed(x)
        out, h = self.gru(embedded)  # out: [b x seq x hid*2] (biRNN)
        return out, h


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=32):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, embedding_layer):
        B = input.size(0)

        embedded = embedding_layer(input).view(1, B, -1)
        embedded = self.dropout(embedded)
        # print(embedded.size(), hidden.size())
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs)

        output = torch.cat((embedded, attn_applied.permute(1, 0, 2)), 2)
        output = self.attn_combine(output)


        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        return result


class Copy_decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=32, summary_len=6):
        super(Copy_decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.summary_len = summary_len

        self.mode_linear = nn.Linear(self.hidden_size * 2, 1)
        self.mode_gate = nn.Sigmoid()

        self.attn = nn.Linear(self.hidden_size * 3, self.max_length)
        self.query = nn.Linear(self.hidden_size * 3, self.max_length)
        self.attn_summary = nn.Linear(self.hidden_size * 2, self.summary_len)
        self.linear_copy = nn.Linear(self.hidden_size * 3, 1)
        self.pointer = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 4, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size * 2)
        self.out = nn.Linear(self.hidden_size * 2, self.output_size)

    def forward(self, input, hidden, encoder_outputs, summary, summary_outputs, embedding_layer):
        B = input.size(0)

        embedded_input = embedding_layer(input).view(1, B, -1)
        embedded_input = self.dropout(embedded_input)

        # embedded_summary = embedding_layer(summary)
        # embedded_summary = self.dropout(embedded_summary)
        # copy_attn_weights = F.softmax(
        #     self.attn_summary(torch.cat((embedded_input[0], hidden[0]), 1)), dim=1)
        # print(p_copy)
        # print(copy_attn_weights)
        # copy_attn_applied = torch.bmm(copy_attn_weights.unsqueeze(1),
        #                               summary_outputs)
        # print(copy_attn_applied.size(),"copy_applied")
        # print(copy_attn_applied.size(), "copy_attn_applied")
        # print(copy_attn_weights.size(), "copy_attn_weights")
        # print(embedded_input.size(), hidden.size(), "embedded, hidden")
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded_input.view(B, -1), hidden.view(B, -1)), 1)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs)

        query = F.softmax(
            self.query(torch.cat((embedded_input.view(B, -1), hidden.view(B, -1)), 1)), dim=1)
        query_applied = torch.bmm(query.unsqueeze(1), summary_outputs)

        output = torch.cat((embedded_input.view(B, -1), attn_applied.view(B, -1), query_applied.view(B, -1)), 1)

        output = self.attn_combine(output)

        output = F.selu(output)
        # print(output.size())
        output, hidden = self.gru(output.view(1, B, -1), hidden)

        # print(attn_applied.size(),encoder_outputs.size())



        # print(query.size(), summary_outputs.size())
        # summary_attn = torch.bmm(query.unsqueeze(1), summary_outputs)
        # print(summary_attn.size())


        # copy_word = F.log_softmax(torch.bmm(summary_outputs, query.unsqueeze(2)), dim=1)


        # output = output.unsqueeze(1)
        # output = output.permute(1, 0, 2)
        # hidden = hidden.permute(1, 0, 2)




        # query = self.query(torch.cat((embedded_input[0], hidden[0]), 1)).unsqueeze(2)
        # copy_word = F.log_softmax(torch.bmm(summary_outputs, query), dim=1)
        # (5,6,1)
        p_copy = self.mode_gate(self.mode_linear(output))
        # p_copy = self.mode_gate(self.linear_copy(torch.cat((embedded_input[0], hidden[0]), 1)))
        copy_word = F.log_softmax(self.pointer(output[0]), dim=1)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return p_copy, copy_word, output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size * 2))
        return result


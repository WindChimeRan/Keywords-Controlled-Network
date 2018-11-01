import torch
from data_utils import Dictionary, Corpus, batch_with_neg_iter
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from models.seq2seq import Encoder_GRU, Summary_decoder, Summary_encoder

from models.copynet import AttnDecoderRNN, Copy_decoder
import numpy as np
import random
import time
import datetime

device = torch.device('cuda:1')


def source_key_eval(embedding_layer, context_encoder, summary_encoder, decoder, pair):

    batch_size =1
    s, summary_s = pair
    s = s.to(device).unsqueeze(0)
    summary_s = summary_s.to(device).unsqueeze(0)
    # p = p.to(device).unsqueeze(0)
    # summary_p = summary_p.to(device).unsqueeze(0)

    s_hidden = context_encoder.init_hidden(batch_size).to(device)
    p_hidden = context_encoder.init_hidden(batch_size).to(device)

    s_summary_hidden_encode = summary_encoder.init_hidden(batch_size).to(device)
    # p_summary_hidden_encode = summary_encoder.init_hidden(batch_size).to(device)

    # print(p.size(), p_hidden.size())
    s_out, s_context, s_encoder_outputs = context_encoder(s, s_hidden, embedding_layer)

    # p_out, p_context, p_encoder_outputs = context_encoder(p, p_hidden, embedding_layer)

    _, s_summary_context, s_summary_hiddens = summary_encoder(summary_s, s_summary_hidden_encode, embedding_layer)
    # _, p_summary_context, p_summary_hiddens = summary_encoder(summary_p, p_summary_hidden_encode, embedding_layer)
    # ------------------------------------------------------------------------------ #

    s2s = evaluate_decode(decoder=decoder, embedding_layer=embedding_layer, decoder_hidden=s_context,
                           target_length=32, encoder_outputs=s_encoder_outputs, summary=summary_s,
                           summary_hiddens=s_summary_hiddens)
    # p2s = evaluate_decode(decoder=decoder, embedding_layer=embedding_layer, decoder_hidden=p_context,
    #                        target_length=32, encoder_outputs=p_encoder_outputs, summary=summary_s,
    # #                        summary_hiddens=s_summary_hiddens)
    # s2p = evaluate_decode(decoder=decoder, embedding_layer=embedding_layer, decoder_hidden=s_context,
    #                        target_length=32, encoder_outputs=s_encoder_outputs, summary=summary_p,
    #                        summary_hiddens=p_summary_hiddens)
    # p2p = evaluate_decode(decoder=decoder, embedding_layer=embedding_layer, decoder_hidden=p_context,
    #                        target_length=32, encoder_outputs=p_encoder_outputs, summary=summary_p,
    #                        summary_hiddens=p_summary_hiddens)

    return s2s, None, None, None


def evaluate_pair(embedding_layer, context_encoder, summary_encoder, decoder, pair):

    batch_size =1
    s, summary_s, p, summary_p = pair
    s = s.to(device).unsqueeze(0)
    summary_s = summary_s.to(device).unsqueeze(0)
    p = p.to(device).unsqueeze(0)
    summary_p = summary_p.to(device).unsqueeze(0)

    s_hidden = context_encoder.init_hidden(batch_size).to(device)
    p_hidden = context_encoder.init_hidden(batch_size).to(device)

    s_summary_hidden_encode = summary_encoder.init_hidden(batch_size).to(device)
    p_summary_hidden_encode = summary_encoder.init_hidden(batch_size).to(device)

    # print(p.size(), p_hidden.size())
    s_out, s_context, s_encoder_outputs = context_encoder(s, s_hidden, embedding_layer)

    p_out, p_context, p_encoder_outputs = context_encoder(p, p_hidden, embedding_layer)

    _, s_summary_context, s_summary_hiddens = summary_encoder(summary_s, s_summary_hidden_encode, embedding_layer)
    _, p_summary_context, p_summary_hiddens = summary_encoder(summary_p, p_summary_hidden_encode, embedding_layer)
    # ------------------------------------------------------------------------------ #

    s2s = evaluate_decode(decoder=decoder, embedding_layer=embedding_layer, decoder_hidden=s_context,
                           target_length=32, encoder_outputs=s_encoder_outputs, summary=summary_s,
                           summary_hiddens=s_summary_hiddens)
    p2s = evaluate_decode(decoder=decoder, embedding_layer=embedding_layer, decoder_hidden=p_context,
                           target_length=32, encoder_outputs=p_encoder_outputs, summary=summary_s,
                           summary_hiddens=s_summary_hiddens)
    s2p = evaluate_decode(decoder=decoder, embedding_layer=embedding_layer, decoder_hidden=s_context,
                           target_length=32, encoder_outputs=s_encoder_outputs, summary=summary_p,
                           summary_hiddens=p_summary_hiddens)
    p2p = evaluate_decode(decoder=decoder, embedding_layer=embedding_layer, decoder_hidden=p_context,
                           target_length=32, encoder_outputs=p_encoder_outputs, summary=summary_p,
                           summary_hiddens=p_summary_hiddens)

    return s2s, s2p, p2p, p2s


def evaluate_decode(decoder, embedding_layer, decoder_hidden, target_length,
                encoder_outputs, summary, summary_hiddens):
    sos = torch.mul(torch.ones((1, 1), dtype=torch.long), corpus.dictionary.word2idx['<s>'])
    decoder_input = Variable(sos).to(device)

    decoded_words = []
    for di in range(target_length):
        p_copy, copy_word, decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs, summary, summary_hiddens, embedding_layer)

        _, copy_topi = copy_word.data.topk(1, dim=1)

        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]

        unfilterred_copy_word = summary.squeeze()[copy_topi.squeeze()]
        # print(summary, copy_topi.squeeze())
        # print(p_copy.size())
        # print(unfilterred_copy_word.size())
        # print(topi.size())
        filterred_copy = torch.where(p_copy.squeeze() > 0.5, unfilterred_copy_word.unsqueeze(0), topi.squeeze(0))

        # print(filterred_copy.size(),'now')
        decoder_input = Variable(filterred_copy)

        if ni == corpus.dictionary.word2idx['</s>']:
            break
        else:
            decoded_words.append(corpus.dictionary.idx2word[filterred_copy.item()])
    return ' '.join(decoded_words)


def decode_loss(decoder, embedding_layer, decoder_hidden, target_length,
                encoder_outputs, summary, summary_hiddens, target_variable, batch_size, teacher_forcing_ratio):
    # decoder_hidden : encoder_last_hidden
    # encoder_outputs: [encoder_hidden...]
    # target_variable:
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # use_teacher_forcing = False
    loss = 0

    sos = torch.mul(torch.ones((batch_size, 1), dtype=torch.long), corpus.dictionary.word2idx['<s>'])
    decoder_input = Variable(sos).to(device)

    # if False:
    if use_teacher_forcing:
        for di in range(target_length):

            p_copy, copy_word, decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, summary, summary_hiddens, embedding_layer)

            target_y = target_variable[:, di] + 0
            # remove sos, eos, pad
            target_y[(target_y == 0) | (target_y == 1) | (target_y == 2)] = -1

            loss_g = F.nll_loss(decoder_output, target_variable[:, di], reduce=False)

            _, target_copy = (target_y.expand_as(summary.t()) == summary.t()).topk(1, dim=0)

            loss_c = F.nll_loss(copy_word.squeeze(), target_copy.squeeze())

            copy_gate_target = (target_y.expand_as(summary.t()) == summary.t()).t().sum(1)
            copy_gate_target[copy_gate_target > 0] = 1

            # print(p_copy.size(), copy_gate_target.float().squeeze())
            loss_copy_gate = F.binary_cross_entropy(p_copy.squeeze(), copy_gate_target.float().squeeze(), reduce=False).squeeze()

            loss_step = torch.mean(torch.mul(copy_gate_target.float(), loss_c) + loss_copy_gate +
                                   torch.mul(1-copy_gate_target.float(), loss_g))
            loss += loss_step

            # topv, topi = copy_word.data.topk(1, dim=1)

            # loss += F.nll_loss(decoder_output, target_variable[:, di])


            decoder_input = target_variable[:, di]  # Teacher forcing

    else:
        for di in range(target_length):
            p_copy, copy_word, decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, summary, summary_hiddens, embedding_layer)


            target_y = target_variable[:, di] + 0
            # remove sos, eos, pad
            target_y[(target_y == 0) | (target_y == 1) | (target_y == 2)] = -1

            loss_g = F.nll_loss(decoder_output, target_variable[:, di], reduce=False)

            _, target_copy = (target_y.expand_as(summary.t()) == summary.t()).topk(1, dim=0)

            loss_c = F.nll_loss(copy_word.squeeze(), target_copy.squeeze())

            copy_gate_target = (target_y.expand_as(summary.t()) == summary.t()).t().sum(1)
            copy_gate_target[copy_gate_target > 0] = 1

            # print(p_copy.size(), copy_gate_target.float().squeeze())
            loss_copy_gate = F.binary_cross_entropy(p_copy.squeeze(), copy_gate_target.float().squeeze(), reduce=False).squeeze()

            loss_step = torch.mean(torch.mul(copy_gate_target.float(), loss_c) + loss_copy_gate +
                                   torch.mul(1-copy_gate_target.float(), loss_g))
            loss += loss_step

            # print(p_copy.squeeze())
            _, copy_topi = copy_word.data.topk(1, dim=1)
            # print(copy_topi.size())
            # print(p_copy.size())
            # loss += F.nll_loss(decoder_output, target_variable[:, di])


            # decoder_input = target_variable[:, di]  # Teacher forcing

            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            # print(topi)
            # print(p_copy.squeeze()>0.5)
            # print(copy_topi)
            # print(summary)
            unfilterred_copy_word = torch.diagonal(summary[:, copy_topi.squeeze()])
            # print(p_copy.squeeze())
            # print(unfilterred_copy_word)
            # print(topi.squeeze())
            # print('----------------')
            generate_copy = torch.where(p_copy.squeeze()>0.5, unfilterred_copy_word, topi.squeeze())
            # print(filterred_copy)
            # topi if p_copy==0 else original
            # print(generate_copy.size())
            # 5
            decoder_input = Variable(generate_copy.unsqueeze(1))

            if ni == corpus.dictionary.word2idx['</s>']:
                break
    return loss

def assist(corpus):
    context_encoder = torch.load('saved_models/context_encoder.pkl').to(device)
    summary_encoder = torch.load('saved_models/summary_encoder.pkl').to(device)
    decoder = torch.load('saved_models/decoder.pkl').to(device)
    embedding_layer = torch.load('saved_models/embedding_layer.pkl').to(device)

    data_iter = corpus.source_keywords(batch_size=1, pretrain=False, roll_back=False)

    with torch.no_grad():
        for i, pair in enumerate(data_iter):
            pair = pair[:2]
            pair = map(lambda x: x[0], pair)
            # e_source, e_summary_source, e_paraphrase, e_summary_paraphrase = pair

            s2s, s2p, p2p, p2s = source_key_eval(embedding_layer=embedding_layer, context_encoder=context_encoder,
                                               summary_encoder=summary_encoder, decoder=decoder, pair=pair)
            result = ' '.join(filter(lambda tok: tok not in ['<s>', '<\s>', '<blank>'], s2s.split())) + '\n'
            print(result)

def eval_mscoco2txt(corpus):
    context_encoder = torch.load('saved_models/context_encoder.pkl').to(device)
    summary_encoder = torch.load('saved_models/summary_encoder.pkl').to(device)
    decoder = torch.load('saved_models/decoder.pkl').to(device)
    embedding_layer = torch.load('saved_models/embedding_layer.pkl').to(device)

    data_iter = corpus.batch_iter(batch_size=1, pretrain=False, roll_back=False)
    start = time.time()
    with open('./data/mscoco/hyps.txt', 'w') as f:
        with torch.no_grad():
            for i, pair in enumerate(data_iter):
                pair = map(lambda x: x[0], pair)
                # e_source, e_summary_source, e_paraphrase, e_summary_paraphrase = pair

                s2s, s2p, p2p, p2s = evaluate_pair(embedding_layer=embedding_layer, context_encoder=context_encoder,
                                                   summary_encoder=summary_encoder, decoder=decoder, pair=pair)
                result = ' '.join(filter(lambda tok: tok not in ['<s>', '<\s>', '<blank>'], s2p.split())) + '\n'
                f.write(result)
                if i % 100 == 0:
                    duration = time.time() - start
                    print(
                        'rate = {:.3f}% \t duration = {:.1f}min'.format(
                            i / corpus.dataset_lines, duration / 60))

def train_iter(corpus):
    LR = 1e-3

    batch_size = 64

    pretrain = False
    input_len = corpus.max_len  # 35

    # dataset_lines = 500000.0 if pretrain else 5370128.0
    # dataset_lines = 662326.0
    dataset_lines = corpus.dataset_lines
    ae_ratio = 0.1

    # batch_size_neg = 2 * batch_size
    dict_size = len(corpus.dictionary.word2idx)
    embedding_layer = nn.Embedding(dict_size, 300).to(device)
    context_encoder = Encoder_GRU(dict_size, embed_size=300, hidden_size=300, num_layers=1).to(device)
    # summary_decoder = Summary_decoder(embed_size=300, hidden_size=300, output_size=summary_len).to(device)
    summary_encoder = Summary_encoder(dict_size, embed_size=300, hidden_size=300, num_layers=1).to(device)
    # copy_decoder = CopyNetDecoder(corpus.dictionary.idx, embed_size=200, hidden_size=300)
    # decoder = AttnDecoderRNN(hidden_size=300, output_size=corpus.dictionary.idx, max_length=input_len)
    decoder = Copy_decoder(hidden_size=300, output_size=dict_size, max_length=input_len).to(device)

    embedding_optimizer = torch.optim.Adam(embedding_layer.parameters(), lr=LR)
    context_optimizer = torch.optim.Adam(context_encoder.parameters(), lr=LR)
    summary_encoder_optimizer = torch.optim.Adam(summary_encoder.parameters(), lr=LR)
    # summary_decoder_optimizer = torch.optim.Adam(summary_decoder.parameters(), lr=LR)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)
    # loss_func = ContrastiveLoss(margin=1)
    data_iter = corpus.batch_iter(batch_size, pretrain=pretrain)
    start_training_time = time.time()
    for step, batch in enumerate(data_iter):

        start_time = time.time()

        s, summary_s, p, summary_p = batch
        s = s.to(device)
        summary_s = summary_s.to(device)
        p = p.to(device)
        summary_p = summary_p.to(device)

        s_hidden = context_encoder.init_hidden(batch_size).to(device)
        p_hidden = context_encoder.init_hidden(batch_size).to(device)

        s_summary_hidden_encode = summary_encoder.init_hidden(batch_size).to(device)
        p_summary_hidden_encode = summary_encoder.init_hidden(batch_size).to(device)

        s_out, s_context, s_encoder_outputs = context_encoder(s, s_hidden, embedding_layer)
        p_out, p_context, p_encoder_outputs = context_encoder(p, p_hidden, embedding_layer)

        _, s_summary_context, s_summary_hiddens = summary_encoder(summary_s, s_summary_hidden_encode, embedding_layer)
        _, p_summary_context, p_summary_hiddens = summary_encoder(summary_p, p_summary_hidden_encode, embedding_layer)

        s_s_loss = decode_loss(decoder=decoder, embedding_layer=embedding_layer, decoder_hidden=s_context,
                             target_length=32, encoder_outputs=s_encoder_outputs, summary=summary_s,
                             summary_hiddens=s_summary_hiddens,
                             target_variable=s, batch_size=batch_size, teacher_forcing_ratio=0.9)

        p_p_loss = decode_loss(decoder=decoder, embedding_layer=embedding_layer, decoder_hidden=p_context,
                             target_length=32, encoder_outputs=p_encoder_outputs, summary=summary_p,
                             summary_hiddens=p_summary_hiddens,
                             target_variable=p, batch_size=batch_size, teacher_forcing_ratio=0.9)

        s_p_loss = decode_loss(decoder=decoder, embedding_layer=embedding_layer, decoder_hidden=s_context,
                             target_length=32, encoder_outputs=s_encoder_outputs, summary=summary_p,
                             summary_hiddens=p_summary_hiddens,
                             target_variable=p, batch_size=batch_size, teacher_forcing_ratio=0.9)

        p_s_loss = decode_loss(decoder=decoder, embedding_layer=embedding_layer, decoder_hidden=p_context,
                             target_length=32, encoder_outputs=p_encoder_outputs, summary=summary_s,
                             summary_hiddens=s_summary_hiddens,
                             target_variable=s, batch_size=batch_size, teacher_forcing_ratio=0.9)



        embedding_optimizer.zero_grad()
        context_optimizer.zero_grad()
        summary_encoder_optimizer.zero_grad()  # clear gradients for this training step
        decoder_optimizer.zero_grad()

        # s_loss.backward()  # backpropagation, compute gradients
        use_ae = True if random.random() < ae_ratio else False
        if use_ae:
            loss = s_s_loss + p_p_loss
        else:
            loss = s_p_loss + p_s_loss
        # loss = s_s_loss + p_p_loss + p_s_loss + s_p_loss

        loss.backward()
        # https://www.reddit.com/r/MachineLearning/comments/31b6x8/gradient_clipping_rnns/
        clip_grad_norm_(embedding_layer.parameters(), 5)
        clip_grad_norm_(context_encoder.parameters(), 5)
        clip_grad_norm_(summary_encoder.parameters(), 5)
        clip_grad_norm_(decoder.parameters(), 5)

        decoder_optimizer.step()
        embedding_optimizer.step()
        context_optimizer.step()  # apply gradients
        # summary_decoder_optimizer.step()
        summary_encoder_optimizer.step()
        duration_step = time.time() - start_time
        if step % 20 == 0:
            duration_training = time.time() - start_training_time

            print("----------------------------------------------------------------------------------------")
            print('step {:d} \t epoch = {:.3f} \t duration = {:.1f}min \t loss = {:.8f}, ({:.3f} sec/step)'.format(step,
                  step * batch_size / dataset_lines, duration_training / 60, loss.item(), duration_step))
            pair = map(lambda x: x[0], next(data_iter))
            e_source, e_summary_source, e_paraphrase, e_summary_paraphrase = pair

            s2s, s2p, p2p, p2s = evaluate_pair(embedding_layer=embedding_layer, context_encoder=context_encoder, summary_encoder=summary_encoder, decoder=decoder, pair=pair)

            e_source = corpus.tensor2seq(e_source)
            e_paraphrase = corpus.tensor2seq(e_paraphrase)
            e_summary_source = corpus.tensor2seq(e_summary_source)
            e_summary_paraphrase = corpus.tensor2seq(e_summary_paraphrase)

            print("source:              \t{}".format(e_source))
            print("gold paraphrase:     \t{}".format(e_paraphrase))
            print("summary source:      \t{}".format(e_summary_source))
            print("summary paraphrase:  \t{}".format(e_summary_paraphrase))
            print("source to source:    \t{}".format(s2s))
            print("source to paraphrase:\t{}".format(s2p))

        if step % 5000 == 0:
            torch.save(context_encoder, 'saved_models/context_encoder.pkl')
            torch.save(summary_encoder,'saved_models/summary_encoder.pkl')
            torch.save(decoder, 'saved_models/decoder.pkl')
            torch.save(embedding_layer, 'saved_models/embedding_layer.pkl')
            print("saved!")
if __name__ == '__main__':

    # DATA = 'PARANMT'
    DATA = 'MSCOCO'
    # MODE = 'TRAIN'
    # MODE = 'TEST'
    MODE = 'CASE'
    TRAIN = False

    if MODE == 'TRAIN':
        if DATA == 'PARANMT':

            corpus = Corpus('./data/para-nmt-5m-processed.txt')
            corpus,dataset_lines = 5370128.0
            # corpus.make_dictionary()
            # corpus.save_dictionary()
            corpus.load_dictionary()
            train_iter(corpus)

        elif DATA == 'MSCOCO':
            corpus = Corpus('./data/mscoco/train.txt')
            corpus.dataset_lines = 662326.0

            # corpus.make_dictionary()
            # corpus.save_dictionary('./data/mscoco_dictionary.pkl')

            corpus.load_dictionary('./data/mscoco_dictionary.pkl')
            train_iter(corpus)
    elif MODE == 'TEST':
        corpus = Corpus('./data/mscoco/test.txt')
        corpus.dataset_lines = 162023.0
        corpus.load_dictionary('./data/mscoco_dictionary.pkl')
        eval_mscoco2txt(corpus)

    elif MODE == 'CASE':
        corpus = Corpus('./data/mscoco/assist.txt')
        corpus.load_dictionary('./data/mscoco_dictionary.pkl')
        assist(corpus)



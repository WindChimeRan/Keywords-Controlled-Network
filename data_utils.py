# coding: utf-8
from torchtext import data, datasets
import torch
import torch.nn.functional as F

import numpy as np
import pke
import itertools
import cPickle
import random
import re


def compose(*funcs):
    """
    Return a new function s.t.
    compose(f,g,...)(x) == f(g(...(x)))
    """

    def inner(data, funcs=funcs):
        result = data
        for f in reversed(funcs):
            result = f(result)
        return result

    return inner


def to_ascii(text):
    return ''.join(map(lambda x:chr(x) if x<128 else ' ', (map(ord, text))))


def keyphrase(input_text):

    extractor = pke.PositionRank()
    extractor.read_text(input_text)
    extractor.candidate_selection()
    extractor.candidate_weighting()
    keyphrases = extractor.get_n_best(n=6, stemming=False)
    return [i[0] for i in keyphrases]

def batch_with_neg_iter(batch):
    """
    return: (s, p, label)
    """
    s, p = batch

    permutation = np.random.permutation(s.shape[0])
    origion = np.arange(s.shape[0])

    s = torch.cat((s[permutation, :], s), 0)
    p = torch.cat((p[permutation, :], p), 0)

    pos_label = torch.Tensor(np.array([True for i in range(s.shape[0])], dtype=int))
    neg_label = torch.Tensor(np.array(permutation == origion, dtype=int))

    label = torch.cat((neg_label, pos_label), 0)
        
    return s, p, label


def merge_mscoco(data='train'):
    try:
        f_source = open('./data/mscoco/%s_source.txt' % data, 'r')
        f_target = open('./data/mscoco/%s_target.txt' % data, 'r')
        f_train = open('./data/mscoco/%s.txt' % data, 'w')
        for num, line_pair in enumerate(zip(f_source, f_target)):
            f_train.write('\t'.join(map(lambda x: x.replace('\n', ''), line_pair)) + '\n')

    finally:
        if f_source:
            f_source.close()
        if f_target:
            f_target.close()
        if f_train:
            f_train.close()


def make_mscoco():
    try:
        f_train = open('./data/mscoco/train.txt', 'r')
        f_test = open('./data/mscoco/test.txt', 'r')
        f = open('./data/mscoco/mscoco.txt', 'w')
        for line in f_train:
            f.write(line)
        for line in f_test:
            f.write(line)

    finally:
        if f_train:
            f_train.close()
        if f_test:
            f_test.close()
        if f:
            f.close()

class Dictionary(object):
    def __init__(self):
        self.word2idx = {'<blank>': 0, '</s>': 1, '<s>': 2, '<unk>': 3}
        self.idx2word = {0: '<blank>', 1: '</s>', 2: '<s>', '<unk>': 3}
        self.word2count = {'<blank>': 0, '</s>': 0, '<s>': 0}
        self.idx = 4

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.word2count[word] = 1
            self.idx += 1
        else:
            self.word2count[word] += 1

    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self, path, max_len=35):
        self.dictionary = Dictionary()
        self.path = path
        self.max_len = max_len
        self.dataset_lines = None

    def make_dictionary(self):

        with open(self.path, 'r') as f:

            for line in f:
                words = re.split('\s+', to_ascii(line)) + ['</s>']
                for word in words:
                    self.dictionary.add_word(word)
        self.save_dictionary()
        print("scan the corpus!")
        self.remove_unk(10)
        print("remove unk words!")
        self.save_dictionary()
        return self.dictionary

    def remove_unk(self, threshold=10):

        for k, v in self.dictionary.word2count.items():
            if v < threshold:
                del self.dictionary.word2count[k]
                # del self.dictionary.idx2word[self.dictionary.word2idx[k]]
                del self.dictionary.word2idx[k]

        cnt = 4
        self.dictionary.idx2word.clear()
        self.dictionary.idx2word = {0: '<blank>', 1: '</s>', 2: '<s>', 3: '<unk>'}
        temp = {'<blank>': 0, '</s>': 1, '<s>': 2, '<unk>': 3}

        for k, v in self.dictionary.word2count.items():
            if k not in ['<blank>', '</s>', '<s>', '<unk>']:
                temp[k] = cnt
                self.dictionary.idx2word[cnt] = k
                cnt += 1
        self.dictionary.word2idx = temp

        return self.dictionary

    def save_dictionary(self, path='./data/dictionary.pkl'):
        cPickle.dump(self.dictionary, open(path, 'wb'))

    def load_dictionary(self, path='./data/dictionary.pkl'):
        self.dictionary = cPickle.load(open(path, 'rb'))

    def tensor2seq(self, tensor):
        return ' '.join(
            filter(lambda x: x != '<blank>', map(lambda x: self.dictionary.idx2word[x], tensor.squeeze().tolist())))

    def batch_iter(self, batch_size=50, pretrain=False, roll_back=True):
        """
        eg:
        1.
        c = 0
        for s, p in corpus.batch_iter(batch_size=4):

        if c > 10:
           break
        2.
        data_iter = corpus.batch_iter(batch_size=4)
        next(data_iter)

        """
        seq2toks = lambda seq: ['<s>'] + seq.split() + ['</s>']

        toks2idx = lambda toks: list(map(lambda tok: self.dictionary.word2idx[tok] if tok in self.dictionary.word2idx else self.dictionary.word2idx['<unk>'], toks))

        pad_len = lambda toks: self.max_len - len(toks)

        pad = lambda idxs: F.pad(torch.LongTensor(idxs), (0, pad_len(idxs)), "constant",
                                 self.dictionary.word2idx['<blank>'])
        join = lambda x: ' '.join(x)

        tokenize = lambda x: x.split()

        rand_diff = lambda x: random.sample(x, len(x)//2 if len(x)//2 < 4 else 4)

        preprocessing = compose(pad, toks2idx, seq2toks)

        # keyphrase_preprocessing = compose(preprocessing, join, keyphrase, to_ascii)
        def keyphrase_preprocessing(line):

            diff_set = lambda a, b: list(set(a.split()) - set(b.split()))

            s, p = map(to_ascii, line.split('\t'))
            s_p = rand_diff(diff_set(s, p))
            p_s = rand_diff(diff_set(p, s))
            key_s = compose(tokenize, join, keyphrase)(s)
            key_p = compose(tokenize, join, keyphrase)(p)
            summary_s = compose(preprocessing, join)([tok for tok in tokenize(s) if tok in s_p or tok in key_s])
            summary_p = compose(preprocessing, join)([tok for tok in tokenize(p) if tok in p_s or tok in key_p])
            return summary_s, summary_p



        sources = torch.LongTensor(batch_size, self.max_len)
        paraphrases = torch.LongTensor(batch_size, self.max_len)

        summary_sources = torch.LongTensor(batch_size, self.max_len)
        summary_paraphrases = torch.LongTensor(batch_size, self.max_len)

        pretrain_threshold = 500000

        with open(self.path, 'r') as f:
            for epoch in itertools.count(1):

                batch_i = 0

                for line_num, line in enumerate(f):
                    # source_summary, paraphrase_summary = map(compose(keyphrase, to_ascii), line.split('\t'))
                    # print(' '.join(source_summary), line.split('\t')[0])
                    # print(line)
                    source, paraphrase = map(compose(preprocessing, to_ascii), line.split('\t'))

                    source_summary, paraphrase_summary = keyphrase_preprocessing(line)

                    sources[batch_i] = source[:]
                    paraphrases[batch_i] = paraphrase[:]
                    summary_sources[batch_i] = source_summary[:]
                    summary_paraphrases[batch_i] = paraphrase_summary[:]
                    batch_i += 1

                    if pretrain and line_num > pretrain_threshold:
                        break
                    if batch_i == batch_size:
                        batch_i = 0
                        yield (sources, summary_sources, paraphrases, summary_paraphrases)
                # finish one epoch
                if roll_back:
                    f.seek(0)

    def source_keywords(self, batch_size=50, pretrain=False, roll_back=True):

        seq2toks = lambda seq: ['<s>'] + seq.split() + ['</s>']

        toks2idx = lambda toks: list(map(lambda tok: self.dictionary.word2idx[tok] if tok in self.dictionary.word2idx else self.dictionary.word2idx['<unk>'], toks))

        pad_len = lambda toks: self.max_len - len(toks)

        pad = lambda idxs: F.pad(torch.LongTensor(idxs), (0, pad_len(idxs)), "constant",
                                 self.dictionary.word2idx['<blank>'])
        join = lambda x: ' '.join(x)

        tokenize = lambda x: x.split()

        rand_diff = lambda x: random.sample(x, len(x)//2 if len(x)//2 < 4 else 4)

        preprocessing = compose(pad, toks2idx, seq2toks)

        # keyphrase_preprocessing = compose(preprocessing, join, keyphrase, to_ascii)
        def keyphrase_preprocessing(line):

            diff_set = lambda a, b: list(set(a.split()) - set(b.split()))

            s, p = map(to_ascii, line.split('\t'))
            s_p = rand_diff(diff_set(s, p))
            p_s = rand_diff(diff_set(p, s))
            key_s = compose(tokenize, join, keyphrase)(s)
            key_p = compose(tokenize, join, keyphrase)(p)
            summary_s = compose(preprocessing, join)([tok for tok in tokenize(s) if tok in s_p or tok in key_s])
            summary_p = compose(preprocessing, join)([tok for tok in tokenize(p) if tok in p_s or tok in key_p])
            return summary_s, summary_p



        sources = torch.LongTensor(batch_size, self.max_len)
        keywords = torch.LongTensor(batch_size, self.max_len)

        summary_sources = torch.LongTensor(batch_size, self.max_len)
        summary_paraphrases = torch.LongTensor(batch_size, self.max_len)

        pretrain_threshold = 500000

        with open(self.path, 'r') as f:
            for epoch in itertools.count(1):

                batch_i = 0

                for line_num, line in enumerate(f):
                    # source_summary, paraphrase_summary = map(compose(keyphrase, to_ascii), line.split('\t'))
                    # print(' '.join(source_summary), line.split('\t')[0])
                    # print(line)
                    # print(line.split('\t'))
                    source, keyword = map(compose(preprocessing, to_ascii), line.split('\t'))

                    # source_summary, paraphrase_summary = keyphrase_preprocessing(line)

                    sources[batch_i] = source[:]
                    keywords[batch_i] = keyword[:]

                    batch_i += 1

                    if pretrain and line_num > pretrain_threshold:
                        break
                    if batch_i == batch_size:
                        batch_i = 0
                        yield (sources, keywords, None, None)
                # finish one epoch
                if roll_back:
                    f.seek(0)
if __name__ == '__main__':
    merge_mscoco('train')
    merge_mscoco('test')
    make_mscoco()

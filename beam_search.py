
# coding: utf-8

import dynet as dy
from collections import defaultdict
import datetime
import math
import heapq

EOS = "<EOS>"
unk = "<UNK>"
beam_size = 1
alpha = 0.6

# In[4]:

def main():

    LAYERS = 2
    INPUT_DIM = 32
    HIDDEN_DIM = 32
    ATTEN_SIZE = 32

    global beam_size

    INPUT_DIM = 512
    HIDDEN_DIM = 512
    ATTEN_SIZE = 256

    BATCH_SIZE = 32

    source = './de.txt'
    target = './en.txt'
    test = './de1.txt'
    blind = './de1.txt'

    source = './train.en-de.low.de'
    target = './train.en-de.low.en'
    test = './valid.en-de.low.de'
    blind = './blind.en-de.low.de'

    enc_dec = EncoderDecoder(LAYERS, INPUT_DIM, HIDDEN_DIM, ATTEN_SIZE, BATCH_SIZE, source, target, test, blind)

    for i in range(1,11):
        beam_size = i
        print datetime.datetime.now()
        enc_dec.translate("beam_" + str(i))

class EncoderDecoder():

    def __init__(self, LAYERS, INPUT_DIM, HIDDEN_DIM, ATTEN_SIZE, BATCH_SIZE, source, target, test, blind):

        self.s_vocab, self.s_id_lookup, self.s_data = self.get_vocab(source)
        self.t_vocab, self.t_id_lookup, self.t_data = self.get_vocab(target)

        self.s_vocab_size = len(self.s_vocab)
        self.t_vocab_size = len(self.t_vocab)

        self.model = dy.Model()

        self.params = {}

        (self.l2r_builder, self.r2l_builder, self.dec_builder, self.params["s_lookup"], self.params["t_lookup"], self.params["W_y"], self.params["b_y"], self.params["W1_att"], self.params["W2_att"], self.params["v_att"]) = self.model.load("4322158.16806")

        self.HIDDEN_DIM = HIDDEN_DIM
        self.BATCH_SIZE = BATCH_SIZE

        self.test_data = self.get_data(test, self.s_vocab)
        # self.blind_data = self.get_data(blind, self.s_vocab)

    def get_vocab(self, train_file):
        vocab = defaultdict(lambda: len(vocab))
        id_lookup = {}
        vocab[EOS]

        train_data = []
        with open(train_file) as f:
            for line in f:
                spl = line.strip().split()

                line_with_id = [vocab[EOS]]
                for word in spl:
                    line_with_id.append(vocab[word])
                line_with_id.append(vocab[EOS])
                train_data.append(line_with_id)

        vocab[unk]
        for k,v in vocab.iteritems():
            id_lookup[v] = k

        return vocab, id_lookup, train_data

    def get_data(self, file_name, vocab):
        data = []
        with open(file_name) as f:
            for line in f:
                spl = line.strip().split()

                line_with_id = [vocab[EOS]]
                for word in spl:
                    if word in vocab:
                        line_with_id.append(vocab[word])
                    else:
                        line_with_id.append(vocab[unk])
                line_with_id.append(vocab[EOS])
                data.append(line_with_id)

        return data

    def attend(self, H_f_mat, h_e, w1dt, source_len, batch_size):
        W2_att = dy.parameter(self.params["W2_att"])
        v_att = dy.parameter(self.params["v_att"])

        w2dt = W2_att*dy.concatenate(list(h_e.s()))

        a_t = dy.transpose(v_att * dy.tanh(dy.colwise_add(w1dt, w2dt)))
        reshaped = a_t
        # reshaped = dy.reshape(a_t, (source_len, ), batch_size)
        a_t_weights = dy.softmax(reshaped)

        attention_c_t = H_f_mat * a_t_weights

        return attention_c_t

    def dict_nlargest(self, d,n):
        return heapq.nlargest(n, d, key=lambda k: d[k])

    def list_nlargest(self, d,n):
        return heapq.nlargest(n, range(len(d)), key=lambda x: d[x])

    # generate from model:
    def generate(self, s_sentence, max_len=150):

        dy.renew_cg()

        global beam_size

        W_y = dy.parameter(self.params["W_y"])
        b_y = dy.parameter(self.params["b_y"])
        s_lookup = self.params["s_lookup"]
        t_lookup = self.params["t_lookup"]

        s_sentence = [self.s_vocab[EOS]] + s_sentence + [self.s_vocab[EOS]]
        s_sentence_rev = list(reversed(s_sentence))

        l2r_state = self.l2r_builder.initial_state()
        r2l_state = self.r2l_builder.initial_state()
        l2r_contexts = []
        r2l_contexts = []

        for cw_l2r in s_sentence:
            l2r_state = l2r_state.add_input(s_lookup[cw_l2r])
            l2r_contexts.append(l2r_state.output())

        for cw_r2l in s_sentence_rev:
            r2l_state = r2l_state.add_input(s_lookup[cw_r2l])
            r2l_contexts.append(r2l_state.output())

        r2l_contexts.reverse()

        H_f = []
        H_f = [dy.concatenate(list(p)) for p in zip(l2r_contexts, r2l_contexts)]

        H_f_mat = dy.concatenate_cols(H_f)
        W1_att = dy.parameter(self.params["W1_att"])
        w1dt = W1_att * H_f_mat

        c_t_init = dy.vecInput(2*self.HIDDEN_DIM)
        # c_t = dy.concatenate([l2r_contexts[-1], r2l_contexts[-1]])

        dec_state_init = self.dec_builder.initial_state()

        possible_list = {("<EOS>", dec_state_init, c_t_init): 0.0}

        for i in range(len(s_sentence)*2):
            t_list = {}

            count_eos = 0

            for (poss, dec_state, c_t), prob in possible_list.iteritems():
                spl_poss = poss.split(' ')

                if i > 1 and spl_poss[-1] == "<EOS>":
                    count_eos += 1
                    t_list[(poss, dec_state, c_t)] = prob
                    continue

                embedding = t_lookup[self.t_vocab[spl_poss[-1]]]

                x_t = dy.concatenate([c_t, embedding])
                dec_state = dec_state.add_input(x_t)
                c_t = self.attend(H_f_mat, dec_state, w1dt, len(s_sentence), 1)
                probs = dy.softmax(W_y*dy.concatenate([c_t, dec_state.output()]) + b_y).vec_value()

                inds = self.list_nlargest(probs, beam_size)

                for ind in inds:
                    sent = poss + " " + self.t_id_lookup[ind]
                    sent_prob = prob + math.log(probs[ind])

                    # lp = (5 + len(sent.split()))/(5+1)

                    # sent_prob = sent_prob/pow(lp, alpha)

                    t_list[(sent, dec_state, c_t)] = sent_prob

            if count_eos == beam_size:
                break

            possible_list = {}

            for tup in self.dict_nlargest(t_list, beam_size):
                possible_list[tup] = t_list[tup]

        final_sent = self.dict_nlargest(possible_list, 1)[0][0]
        return " ".join(final_sent.replace("<EOS>", "").strip().split())


    def translate(self, i):
        f = open(str(i) + '_test.txt', 'w')
        for sent in self.test_data:
            f.write(self.generate(sent))
            f.write('\n')
        f.close()

        # f = open(str(i) + '_blind.txt', 'w')
        # for sent in self.blind_data:
        #     f.write(self.generate(sent))
        #     f.write('\n')
        # f.close()


if __name__ == '__main__':
    main()

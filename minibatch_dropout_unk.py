
# coding: utf-8

import dynet as dy
from collections import defaultdict
import random
import math

EOS = "<EOS>"
unk = "<UNK>"
dropout = 0.2
threshold = 1

# In[4]:

def main():

    LAYERS = 2
    INPUT_DIM = 32
    HIDDEN_DIM = 32
    ATTEN_SIZE = 32

    INPUT_DIM = 512
    HIDDEN_DIM = 512
    ATTEN_SIZE = 256

    BATCH_SIZE = 32

    source = './de1.txt'
    target = './en1.txt'
    test = './de1.txt'
    blind = './de1.txt'

    source = './train.en-de.low.de'
    target = './train.en-de.low.en'
    source_val = './valid.en-de.low.de'
    target_val = './valid.en-de.low.en'
    test = './test.en-de.low.de'
    blind = './blind.en-de.low.de'

    enc_dec = EncoderDecoder(LAYERS, INPUT_DIM, HIDDEN_DIM, ATTEN_SIZE, BATCH_SIZE, source, target, source_val, target_val, test, blind)

    print "Starting train"
    for i in range(30):
        print "Starting epoch: " + str(i)
        enc_dec.train()
    enc_dec.translate(i)


class EncoderDecoder():

    def __init__(self, LAYERS, INPUT_DIM, HIDDEN_DIM, ATTEN_SIZE, BATCH_SIZE, source, target, source_val, target_val, test, blind):

        self.s_vocab, self.s_id_lookup, self.s_data = self.get_vocab(source)
        self.t_vocab, self.t_id_lookup, self.t_data = self.get_vocab(target)

        self.s_vocab_size = len(self.s_vocab)
        self.t_vocab_size = len(self.t_vocab)

        # self.model = dy.Model()

        # self.l2r_builder = dy.LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, self.model)
        # self.r2l_builder = dy.LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, self.model)
        # self.dec_builder = dy.LSTMBuilder(LAYERS, INPUT_DIM+(HIDDEN_DIM*2), HIDDEN_DIM, self.model)

        # self.l2r_builder.set_dropout(dropout)
        # self.r2l_builder.set_dropout(dropout)
        # self.dec_builder.set_dropout(dropout)

        # self.params = {}

        # self.params["s_lookup"] = self.model.add_lookup_parameters((self.s_vocab_size, INPUT_DIM))
        # self.params["t_lookup"] = self.model.add_lookup_parameters((self.t_vocab_size, INPUT_DIM))
        # self.params["W_y"] = self.model.add_parameters((self.t_vocab_size, HIDDEN_DIM*3))
        # self.params["b_y"] = self.model.add_parameters((self.t_vocab_size))

        # self.params["W1_att"] = self.model.add_parameters((ATTEN_SIZE, 2*HIDDEN_DIM))
        # self.params["W2_att"] = self.model.add_parameters((ATTEN_SIZE, LAYERS*HIDDEN_DIM*2))
        # self.params["v_att"] = self.model.add_parameters((1, ATTEN_SIZE))

        self.model = dy.Model()

        self.params = {}

        (self.l2r_builder, self.r2l_builder, self.dec_builder, self.params["s_lookup"], self.params["t_lookup"], self.params["W_y"], self.params["b_y"], self.params["W1_att"], self.params["W2_att"], self.params["v_att"]) = self.model.load("18.2164769676")

        self.l2r_builder.set_dropout(dropout)
        self.r2l_builder.set_dropout(dropout)
        self.dec_builder.set_dropout(dropout)

        self.HIDDEN_DIM = HIDDEN_DIM
        self.BATCH_SIZE = BATCH_SIZE

        self.s_val_data = self.get_data(source_val, self.s_vocab)
        self.t_val_data = self.get_data(target_val, self.t_vocab)
        self.test_data = self.get_data(test, self.s_vocab)
        self.blind_data = self.get_data(blind, self.s_vocab)

        self.max_perp = 1000000.0

    def get_vocab(self, train_file):
        vocab = defaultdict(lambda: len(vocab))
        id_lookup = {}
        vocab[EOS]
        vocab[unk]

        freqs = {}

        with open(train_file) as f:
            for line in f:
                spl = line.strip().split()

                for word in spl:
                    if word in freqs:
                        vocab[word]
                    else:
                        freqs[word] = True

        train_data = []

        with open(train_file) as f:
            for line in f:
                spl = line.strip().split()

                line_with_id = [vocab[EOS]]
                for word in spl:
                    if word in vocab:
                        line_with_id.append(vocab[word])
                    else:
                        line_with_id.append(vocab[unk])
                line_with_id.append(vocab[EOS])
                train_data.append(line_with_id)

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

    def make_batches(self):
        z = zip(self.s_data, self.t_data)
        z.sort(key=lambda (x,y): -len(x))

        batches = []

        cur_batch = []
        cur_len = len(z[0][0])

        for pair in z:
            s_len = len(pair[0])

            if len(cur_batch) >= self.BATCH_SIZE or s_len != cur_len:
                batches.append(cur_batch)
                cur_batch = []
                cur_len = s_len

            cur_batch.append(pair)

        if len(cur_batch) > 0:
            batches.append(cur_batch)
        return batches

    def make_batches_val(self):
        z = zip(self.s_val_data, self.t_val_data)
        z.sort(key=lambda (x,y): -len(x))

        batches = []

        cur_batch = []
        cur_len = len(z[0][0])

        for pair in z:
            s_len = len(pair[0])

            if len(cur_batch) >= self.BATCH_SIZE or s_len != cur_len:
                batches.append(cur_batch)
                cur_batch = []
                cur_len = s_len

            cur_batch.append(pair)

        if len(cur_batch) > 0:
            batches.append(cur_batch)
        return batches

    def get_val_perp(self):
        batches = self.make_batches_val()

        total_loss = 0.0
        total_words = 0

        random.shuffle(batches)

        for batch in batches:
            loss, words = self.calculate_batch_loss(batch)
            total_loss += loss.value()
            total_words += words

        print total_words
        perp = math.exp(total_loss/total_words)
        print "Val perp: " + str(perp)

        if perp < self.max_perp:
            self.model.save(str(perp), [self.l2r_builder, self.r2l_builder, self.dec_builder, self.params["s_lookup"], self.params["t_lookup"], self.params["W_y"], self.params["b_y"], self.params["W1_att"], self.params["W2_att"], self.params["v_att"]])
            self.max_perp = perp

        return perp

    def train(self):
        trainer = dy.SimpleSGDTrainer(self.model)
        batches = self.make_batches()
        num_examples = 0
        total_loss = 0.0
        total_words = 0

        batch_count = 0

        random.shuffle(batches)

        for batch in batches:

            num_examples += len(batch)
            loss, words = self.calculate_batch_loss(batch)
            total_loss += loss.value()
            total_words += words
            loss.backward()
            trainer.update()

            if batch_count % 16 == 0:
                print math.exp(total_loss/total_words)
            if batch_count % 640 == 0:
                self.get_val_perp()

            batch_count += 1

        print
        f = open('training_perp.txt', 'a')
        f.write(str(total_loss/total_words) + '\n')
        f.close()

        val_perp = self.get_val_perp()
        f = open('val_perp.txt', 'a')
        f.write(str(val_perp) + '\n')
        f.close()

    def calculate_batch_loss(self, batch):
        dy.renew_cg()

        W_y = dy.parameter(self.params["W_y"])
        b_y = dy.parameter(self.params["b_y"])
        s_lookup = self.params["s_lookup"]
        t_lookup = self.params["t_lookup"]

        s_batch = [x[0] for x in batch]
        t_batch = [x[1] for x in batch]

        wids = []

        for i in range(len(s_batch[0])):
            wids.append([sent[i] for sent in s_batch])

        wids_rev = list(reversed(wids))

        l2r_state = self.l2r_builder.initial_state()
        r2l_state = self.r2l_builder.initial_state()
        l2r_contexts = []
        r2l_contexts = []

        for wid in wids:
            l2r_state = l2r_state.add_input(dy.lookup_batch(s_lookup, wid))
            l2r_contexts.append(l2r_state.output())

        for wid in wids_rev:
            r2l_state = r2l_state.add_input(dy.lookup_batch(s_lookup, wid))
            r2l_contexts.append(r2l_state.output())

        r2l_contexts.reverse()

        losses = []

        H_f = []
        H_f = [dy.concatenate(list(p)) for p in zip(l2r_contexts, r2l_contexts)]

        H_f_mat = dy.concatenate_cols(H_f)
        W1_att = dy.parameter(self.params["W1_att"])
        w1dt = W1_att * H_f_mat

        t_wids = []
        masks = []

        num_words = 0

        for i in range(len(t_batch[0])):
            t_wids.append([(sent[i] if len(sent) > i else self.t_vocab[EOS]) for sent in t_batch])
            mask = [(1 if len(sent) > i else 0) for sent in t_batch]
            masks.append(mask)
            num_words += sum(mask)

        c_t = dy.vecInput(2*self.HIDDEN_DIM)

        words = [self.t_vocab[EOS]] * len(t_batch)
        embedding = dy.lookup_batch(t_lookup, words)

        dec_state = self.dec_builder.initial_state()

        for t_wid, mask in zip(t_wids, masks):
            x_t = dy.concatenate([c_t, embedding])
            dec_state = dec_state.add_input(x_t)

            c_t = self.attend(H_f_mat, dec_state, w1dt, len(s_batch[0]), len(wids[0]))

            probs = dy.affine_transform([b_y, W_y, dy.concatenate([c_t, dec_state.output()])])
            loss = dy.pickneglogsoftmax_batch(probs, t_wid)

            if mask[-1] != 1:
                mask_expr = dy.inputVector(mask)
                mask_expr = dy.reshape(mask_expr, (1,), len(t_batch))
                loss = loss * mask_expr

            losses.append(loss)
            embedding = dy.lookup_batch(t_lookup, t_wid)

        loss = dy.sum_batches(dy.esum(losses))  # /len(wids[0])
        return loss, num_words

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

    # generate from model:
    def generate(self, s_sentence, max_len=150):

        dy.renew_cg()

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

        c_t = dy.vecInput(2*self.HIDDEN_DIM)
        embedding = t_lookup[self.t_vocab["<EOS>"]]

        dec_state = self.dec_builder.initial_state()

        t_sentence = []

        count_eos = 0

        for i in range(len(s_sentence)*2):
            if count_eos == 2:
                break

            x_t = dy.concatenate([c_t, embedding])
            dec_state = dec_state.add_input(x_t)

            c_t = self.attend(H_f_mat, dec_state, w1dt, len(s_sentence), 1)
            probs = dy.softmax(W_y*dy.concatenate([c_t, dec_state.output()]) + b_y).vec_value()
            word = probs.index(max(probs))

            embedding = t_lookup[word]

            if self.t_id_lookup[word] == "<EOS>":
                count_eos += 1
                continue

            t_sentence.append(self.t_id_lookup[word])

        return " ".join(t_sentence)

    def translate(self, i):
        f = open(str(i) + '_test.txt', 'w')
        for sent in self.test_data:
            f.write(self.generate(sent))
            f.write('\n')
        f.close()

        f = open(str(i) + '_blind.txt', 'w')
        for sent in self.blind_data:
            f.write(self.generate(sent))
            f.write('\n')
        f.close()


if __name__ == '__main__':
    main()

import os
import sys

import tensorflow as tf
from tensorflow.contrib import crf

from crf import LinearCRF
from data_utils import *

embedding_size = 200
hidden_size = 256
embedding_keep_prob = 0.8
rnn_keep_prob = 0.5
fc_keep_prob = 0.6
batch_size = 64
lr = 0.001
epochs = 30

word_path = 'data/word_vocab.pkl'
tag_path = 'data/tag_vocab.pkl'

use_gpu = True
os.environ['CUDA_VISIBLE_DEVICES'] = str(1 - int(use_gpu))


class BiLSTM_CRF:
    """BiLSTM-CRF for Chinese NER task.
    """

    def __init__(self):
        # place holder
        self.word_indices = tf.placeholder(tf.int32, shape=[None, None])
        self.tag_indices = tf.placeholder(tf.int32, shape=[None, None])
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None])
        self.embedding_keep_prob = tf.placeholder(tf.float32, shape=[])
        self.rnn_keep_prob = tf.placeholder(tf.float32, shape=[])
        self.fc_keep_prob = tf.placeholder(tf.float32, shape=[])

        # words and tags vocabulary
        self.word_vocab, self.id2word = read_vocab(word_path)
        self.vocab_size = len(self.word_vocab)
        self.tag_vocab, self.id2tag = read_vocab(tag_path)
        self.num_tags = len(self.tag_vocab)

        # embedding layer, randomly initializing char vectors
        self.embedding = tf.Variable(tf.random_normal([
            self.vocab_size, embedding_size], stddev=0.1, dtype=tf.float32))
        # shape = (batch_size, max_seq_len, embedding_size)
        self.embed_x = tf.nn.embedding_lookup(self.embedding, self.word_indices)
        self.embed_x = tf.nn.dropout(self.embed_x, self.embedding_keep_prob)

        # rnn layer
        if use_gpu:
            cell = tf.keras.layers.CuDNNGRU(hidden_size, return_sequences=True)
        else:
            cell = tf.keras.layers.GRU(hidden_size, return_sequences=True)
        # shape = (batch_size, max_seq_len, hidden_size)
        self.rnn_output = tf.keras.layers.Bidirectional(cell)(self.embed_x)
        self.rnn_output = tf.nn.dropout(self.rnn_output, self.rnn_keep_prob)

        # fully connection layer
        # shape = (batch_size, max_seq_len, num_tags)
        self.logits = tf.keras.layers.Dense(self.num_tags)(self.rnn_output)
        self.logits = tf.nn.dropout(self.logits, self.fc_keep_prob)

        # crf loss
        # log_likelihood, self.transition_params = crf.crf_log_likelihood(
        #     self.logits, self.tag_indices, self.sequence_lengths)
        crf_model = LinearCRF(self.logits, self.tag_indices, self.sequence_lengths)
        self.log_likelihood, self.transition_params = crf_model.log_likelihood()
        self.loss = tf.reduce_mean(-self.log_likelihood)

        # training operation
        self.global_step = tf.Variable(0, trainable=False)
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss, self.global_step)

    def get_feed_dict(self, xs, ys, is_training=True):
        """Given some data, pad it and build a feed dictionary.
        If not training, set the dropout keep probability to 1.0
        """
        xs_indices = [words2id(words, self.word_vocab) for words in xs]
        xs_pad, sequence_lengths = pad_sequences(xs_indices)
        feed_dict = {
            self.word_indices: xs_pad,
            self.sequence_lengths: sequence_lengths,
            self.embedding_keep_prob: embedding_keep_prob,
            self.rnn_keep_prob: rnn_keep_prob,
            self.fc_keep_prob: fc_keep_prob}

        if ys:
            ys_indices = [tags2id(tags, self.tag_vocab) for tags in ys]
            ys_pad, _ = pad_sequences(ys_indices)
            feed_dict[self.tag_indices] = ys_pad

        if not is_training:
            feed_dict[self.embedding_keep_prob] = 1.0
            feed_dict[self.rnn_keep_prob] = 1.0
            feed_dict[self.fc_keep_prob] = 1.0
        return feed_dict

    def predict_batch(self, sess, xs, ys=None):
        """Predict batch tags, return tuple of batch loss and tags.
        If ys is not None, calculate and return the loss value of xs and ys,
        otherwise loss value is None.
        """
        feed_dict = self.get_feed_dict(xs, ys, is_training=False)
        if ys:
            loss = sess.run(self.loss, feed_dict=feed_dict)
        else:
            loss = None
        lengths = feed_dict[self.sequence_lengths]
        logits, transition_params = sess.run(
            [self.logits, self.transition_params], feed_dict=feed_dict)
        tags = []
        for i, (x, seq_len, logit) in enumerate(zip(xs, lengths, logits)):
            seq_indices = crf.viterbi_decode(logit[:seq_len], transition_params)[0]
            seq = [self.id2tag[i] for i in seq_indices]
            tags.append(seq)
            # print(i, x, '\n', y, '\n', tag_predict, '\n')
        return loss, tags

    def train(self, train_data, test_data):
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(epochs):
                print('on epoch: %s' % (epoch + 1))
                batches = batch_yield(train_data, batch_size, shuffle=True)
                for step, (xs, ys) in enumerate(batches):
                    # 转换batch花费很多CPU，导致GPU利用低
                    sess.run(self.train_op, feed_dict=self.get_feed_dict(xs, ys))
                    if step % 20 == 0:
                        loss = self.predict_batch(sess, xs, ys)[0]
                        print('loss value: %.4f' % loss)

                self.test(sess, test_data)

            saver.save(sess, 'checkpoints/model.ckpt', self.global_step)

    def test(self, sess, test_data):
        """Using test dataset to evaluate the model.
        """
        total_loss, step = 0.0, 0
        batches = batch_yield(test_data, batch_size, shuffle=True)
        for step, (xs, ys) in enumerate(batches):
            loss = self.predict_batch(sess, xs, ys)[0]
            total_loss += loss
        print('\nevaluation average loss: %.4f\n' % (total_loss / (step + 1)))

    def predict(self, sentences=None, online=True):
        """Predict tags of sentences (list of sentence).
        """
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

            if not online:
                data = [(list(s), None) for s in sentences]
                tags = []
                batches = batch_yield(data, batch_size)
                for xs, _ in batches:
                    tags_ = self.predict_batch(sess, xs)[1]
                    tags.extend(tags_)
                return tags

            while online:
                sys.stdout.write('Please input your sentence:\n>> ')
                try:
                    words = list(input().strip())
                except KeyboardInterrupt:
                    print('See you next time!')
                if not words:
                    continue
                tags = self.predict_batch(sess, [words])[1][0]
                tagged = ' '.join('%s/%s' % i for i in zip(words, tags))
                print(tagged, '\n')


if __name__ == '__main__':
    model = BiLSTM_CRF()
    if len(sys.argv) > 1 and sys.argv[1].strip() == 'train':
        # checkpoints training
        train_data = read_corpus('data/train.data')
        test_data = read_corpus('data/test.data')
        model.train(train_data, test_data)
    else:
        model.predict(online=True)

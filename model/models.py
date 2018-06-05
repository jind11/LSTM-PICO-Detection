import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix


from .data_utils import minibatches, pad_sequences, get_chunks
from .general_utils import Progbar
from .base_model import BaseModel


class HANNModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config):
        super(HANNModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}
        self.idx_to_words = {idx: word for word, idx in
                           self.config.vocab_words.items()}
        # self.class_weights = [self.config.weight_tags[tag] for idx, tag in sorted(self.idx_to_tag.items())]
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.config.l2_reg_lambda)


    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size)
        self.document_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="document_lengths")

        # shape = (batch size, max length of documents in batch (how many sentences in one abstract), max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="word_ids")

        # shape = (batch_size, max_length of sentence)
        self.sentence_lengths = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_lengths")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                        name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")


    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data
        _, document_lengths = pad_sequences(words, pad_tok=0, nlevels=1)
        word_ids, sentence_lengths = pad_sequences(words, pad_tok=0, nlevels=2)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.document_lengths: document_lengths,
            self.sentence_lengths: sentence_lengths
        }

        if labels is not None:
            labels, _ = pad_sequences(labels, 0, nlevels=1)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, document_lengths


    def add_word_embeddings_op(self):
        """Defines self.word_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids, name="word_embeddings")

        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)


    def add_logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        s = tf.shape(self.word_embeddings)
        word_embeddings_dim = self.config.dim_word

        sentence_lengths = tf.reshape(self.sentence_lengths, shape=[s[0]*s[1]])
        
        word_embeddings = tf.reshape(self.word_embeddings, 
                            shape=[s[0]*s[1], s[-2], word_embeddings_dim])

        with tf.variable_scope("bi-lstm-sentence"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm_sentence)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm_sentence)

            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, word_embeddings,
                    sequence_length=sentence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)

            W_word = tf.get_variable("weight", dtype=tf.float32, 
                    initializer=self.initializer, regularizer=self.regularizer,
                    shape=[2*self.config.hidden_size_lstm_sentence, self.config.attention_size])
            b_word = tf.get_variable("bias", shape=[self.config.attention_size],
                dtype=tf.float32, initializer=tf.zeros_initializer())
            U_word = tf.get_variable("U-noreg", dtype=tf.float32, 
                    initializer=self.initializer, 
                    shape=[self.config.attention_size, 1])

            output = tf.reshape(output, shape=[-1, 2*self.config.hidden_size_lstm_sentence])
            U_sent = tf.tanh(tf.matmul(output, W_word) + b_word)
            A = tf.nn.softmax(tf.reshape(tf.squeeze(tf.matmul(U_sent, U_word)), shape=[-1, s[2]]))
            output = tf.reshape(output, shape=[-1, s[2], 2*self.config.hidden_size_lstm_sentence])
            output = tf.reduce_sum(tf.multiply(output, tf.tile(tf.expand_dims(A, axis=-1), 
                                    [1, 1, 2*self.config.hidden_size_lstm_sentence])), axis=1)

        # dropout
        output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            W_infer = tf.get_variable("weight", dtype=tf.float32, 
                    initializer=self.initializer, regularizer=self.regularizer,
                    shape=[2*self.config.hidden_size_lstm_sentence, self.config.ntags])

            b_infer = tf.get_variable("bias", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            pred = tf.matmul(output, W_infer) + b_infer
            self.logits = tf.reshape(pred, [-1, s[1], self.config.ntags])


    def add_pred_op(self):
        """Defines self.labels_pred

        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With theCRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        if not self.config.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                    tf.int32)


    def add_loss_op(self):
        """Defines the loss"""
        if self.config.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, self.labels, self.document_lengths)
            self.trans_params = trans_params # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.labels)
            # class_weights = tf.constant(self.class_weights)
            # weights = tf.gather(class_weights, self.labels, axis=-1)
            # losses = tf.losses.sparse_softmax_cross_entropy(
            #                         labels=self.labels, 
            #                         logits=self.logits, 
            #                         weights=weights,
            #                         reduction=tf.losses.Reduction.NONE)
            mask = tf.sequence_mask(self.document_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        # add l2 regularizationtete
        l2 = self.config.l2_reg_lambda * sum([
            tf.nn.l2_loss(tf_var)
            for tf_var in tf.trainable_variables()
            if not ("noreg" in tf_var.name or "bias" in tf_var.name)])
        self.loss += l2

        # for tensorboard
        tf.summary.scalar("loss", self.loss)


    def build(self):
        # NER specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init


    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            document_length

        """
        fd, document_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                    [self.logits, self.trans_params], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, document_length in zip(logits, document_lengths):
                logit = logit[:document_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, document_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, document_lengths


    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                    self.config.dropout)

            _, train_loss, summary = self.sess.run(
                    [self.train_op, self.loss, self.merged], feed_dict=fd)

            if not self.config.train_accuracy:
                prog.update(i + 1, [("train loss", train_loss)])
            else:
                labels_pred, document_lengths = self.predict_batch(words)
                accs = []
                for lab, lab_pred, length in zip(labels, labels_pred,
                                                 document_lengths):
                    lab      = lab[:length]
                    lab_pred = lab_pred[:length]
                    accs    += [a==b for (a, b) in zip(lab, lab_pred)]
                acc = np.mean(accs)
                prog.update(i + 1, [("train loss", train_loss), ("accuracy", acc)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        metrics = self.run_evaluate(dev, report=True)
        msg = " - ".join(["{} {:04.3f}".format(k, v)
                    if k == 'acc' else '{} {}'.format(k, ', '.join(['{}: {:04.2f}'.format(a, b) \
                    for a, b in v.items()])) for k, v in metrics.items()])
        self.logger.info(msg)

        return np.mean(list(metrics["f1"].values()))


    def run_evaluate(self, test, report=False):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        accs = []
        labs = []
        labs_pred = []
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, document_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             document_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                # lab_chunks      = set(get_chunks(lab, self.config.vocab_tags))
                # lab_pred_chunks = set(get_chunks(lab_pred,
                                                 # self.config.vocab_tags))

                # correct_preds += len(accs)
                # total_preds   += len(lab_pred)
                # total_correct += len(lab)

                labs.extend(lab)
                labs_pred.extend(lab_pred)

        # p   = correct_preds / total_preds if correct_preds > 0 else 0
        # r   = correct_preds / total_correct if correct_preds > 0 else 0
        # f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        precision, recall, f1, _ = precision_recall_fscore_support(labs, labs_pred)
        acc = np.mean(accs)

        if report:
            target_names = [self.idx_to_tag[i] for i in range(len(self.idx_to_tag))]
            print(classification_report(labs, labs_pred, target_names=target_names, digits=4))
            print(self.idx_to_tag)
            print(confusion_matrix(labs, labs_pred))

        return {"acc": 100*acc, 
                'precision': {tag: precision[self.config.vocab_tags[tag]] for tag in ['P', 'I', 'O']},
                'recall': {tag: recall[self.config.vocab_tags[tag]] for tag in ['P', 'I', 'O']},
                'f1': {tag: f1[self.config.vocab_tags[tag]] for tag in ['P', 'I', 'O']}}


    def predict(self, words_raw):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds

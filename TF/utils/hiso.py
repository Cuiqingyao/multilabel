"""
Utility functions for constructing MLC models.
multilabel loss functions see：
http://d0evi1.com/sklearn/model_evaluation/ and http://www.jos.org.cn/html/2014/9/4634.htm
"""
import tensorflow as tf
from keras.layers import (GRU, Activation, ActivityRegularization,
                          BatchNormalization, Dense, Dropout, Embedding,
                          concatenate, multiply)
from keras.layers.wrappers import Bidirectional
from keras.objectives import binary_crossentropy
from keras.regularizers import l2


class HISO(object):
    def __init__(self, params, mask_zero=True):
        # input words
        self.wds = tf.placeholder(
            tf.float32, [None, params['words']['dim']], name='words')
        # input pos
        self.pos = tf.placeholder(
            tf.float32, [None, params['pos']['dim']], name='pos')
        # output Y0
        self.Y0 = tf.placeholder(
            tf.float32, [None, params['Y0']['dim']], name='Y0')
        # output Y1
        self.Y1 = tf.placeholder(
            tf.float32, [None, params['Y1']['dim']], name='Y1')

        # 1.base layers: embedding
        wd_embedding = Embedding(
            output_dim=params['embed_size'],
            input_dim=params['voc_size'],
            input_length=params['words']['dim'],
            mask_zero=mask_zero,
            name='wd_embedding')(self.wds)
        # wd_embedding = BatchNormalization(momentum=0.9, name='wd_embedding_BN')(wd_embedding)

        pos_embedding = Embedding(
            output_dim=params['embed_size'],
            input_dim=params['pos_size'],
            input_length=params['pos']['dim'],
            mask_zero=mask_zero,
            name='pos_embedding')(self.pos)
        # pos_embedding = BatchNormalization(momentum=0.9, name='pos_embedding_BN')(pos_embeding)

        # 2. semantic layers: Bidirectional GRU
        wd_Bi_GRU = Bidirectional(
            GRU(params['words']['RNN']['cell'],
                dropout=params['words']['RNN']['drop_out'],
                recurrent_dropout=params['words']['RNN']['rnn_drop_out']),
            merge_mode='concat',
            name='word_Bi_GRU')(wd_embedding)
        if 'batch_norm' in params['words']['RNN']:
            wd_Bi_GRU = BatchNormalization(
                momentum=params['words']['RNN']['batch_norm'],
                name='word_Bi_GRU_BN')(wd_Bi_GRU)

        pos_Bi_GRU = Bidirectional(
            GRU(params['pos']['RNN']['cell'],
                dropout=params['pos']['RNN']['drop_out'],
                recurrent_dropout=params['pos']['RNN']['rnn_drop_out']),
            merge_mode='concat',
            name='word_Bi_GRU')(pos_embedding)
        if 'batch_norm' in params['pos']['RNN']:
            pos_Bi_GRU = BatchNormalization(
                momentum=params['pos']['RNN']['batch_norm'],
                name='pos_Bi_GRU_BN')(pos_Bi_GRU)

        # use pos as attention
        attention_probs = Dense(
            2*params['pos']['RNN']['cell'],
            activation='softmax',
            name='attention_vec')(pos_Bi_GRU)
        attention_mul = multiply(
            [wd_Bi_GRU, attention_probs], name='attention_mul')
        # ATTENTION PART FINISHES HERE

        # 3. middle layer for predict Y0
        kwargs = params['Y0']['kwargs'] if 'kwargs' in params['Y0'] else {}
        if 'W_regularizer' in kwargs:
            kwargs['W_regularizer'] = l2(kwargs['W_regularizer'])
        self.Y0_probs = Dense(
            params['Y0']['dim'],
            # activation='softmax',
            name='Y0_probs',
            bias_regularizer=l2(0.01),
            **kwargs)(pos_Bi_GRU)
        # batch_norm
        if 'batch_norm' in params['Y0']:
            self.Y0_probs = BatchNormalization(
                **params['Y0']['batch_norm'])(self.Y0_probs)
        self.Y0_probs = Activation(params['Y0']['activate_func'])(
            self.Y0_probs)

        if 'activity_reg' in params['Y0']:
            self.Y0_probs = ActivityRegularization(
                name='Y0_activity_reg',
                **params['Y0']['activity_reg'])(self.Y0_probs)

        # 4. upper hidden layers
        # Firstly, learn a hidden layer from Bi_GRU
        # Secondly, consider Y0_preds as middle feature and combine it with hidden layer

        combine_layer = concatenate(
            [self.Y0_probs, attention_mul], axis=-1, name='combine_layer')

        hidden_layer = Dense(
            params['H']['dim'], name='hidden_layer')(combine_layer)
        if 'batch_norm' in params['H']:
            hidden_layer = BatchNormalization(
                momentum=0.9, name='hidden_layer_BN')(hidden_layer)
        hidden_layer = Activation('relu')(hidden_layer)
        if 'drop_out' in params['H']:
            hidden_layer = Dropout(
                params['H']['drop_out'],
                name='hidden_layer_dropout')(hidden_layer)

        # 5. layer for predict Y1
        kwargs = params['Y1']['kwargs'] if 'kwargs' in params['Y1'] else {}
        if 'W_regularizer' in kwargs:
            kwargs['W_regularizer'] = l2(kwargs['W_regularizer'])
        self.Y1_probs = Dense(
            params['Y1']['dim'],
            # activation='softmax',
            name='Y1_probs',
            bias_regularizer=l2(0.01),
            **kwargs)(hidden_layer)
        # batch_norm
        if 'batch_norm' in params['Y1']:
            self.Y1_probs = BatchNormalization(
                **params['Y1']['batch_norm'])(self.Y1_probs)
        self.Y1_probs = Activation(params['Y1']['activate_func'])(
            self.Y1_probs)

        if 'activity_reg' in params['Y1']:
            self.Y1_probs = ActivityRegularization(
                name='Y1_activity_reg',
                **params['Y1']['activity_reg'])(self.Y1_probs)

        # 6. Calculate loss
        with tf.name_scope('loss'):
            Y0_loss = tf.reduce_mean(
                binary_crossentropy(self.Y0, self.Y0_probs), name='Y0_loss')
            Y1_loss = tf.reduce_mean(
                binary_crossentropy(self.Y1, self.Y1_probs), name='Y1_loss')
            self.loss = tf.add_n([Y0_loss, Y1_loss], name='loss')

        self.train_op = tf.train.RMSPropOptimizer(
            params['learning_rate']).minimize(self.loss)

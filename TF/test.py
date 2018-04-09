import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, multiply
from keras.metrics import binary_accuracy
from keras.objectives import binary_crossentropy

np.random.seed(1337)  # for reproducibility

input_dim = 32


def build_model():
    inputs = tf.placeholder(tf.float32, shape=[None, input_dim])
    tf.summary.histogram('inputs', inputs)

    with tf.name_scope('attention_layer'):
        # ATTENTION PART STARTS HERE
        with tf.name_scope('weights'):
            attention_probs = Dense(
                input_dim, activation='softmax', name='attention_vec')(inputs)
            variable_summaries(attention_probs)
        with tf.name_scope('inputs_weighted'):
            attention_mul = multiply([inputs, attention_probs])
            variable_summaries(attention_mul)
            # ATTENTION PART FINISHES HERE

    attention_mul = Dense(64)(attention_mul)
    with tf.name_scope('predictions'):
        preds = Dense(1, activation='sigmoid')(attention_mul)
        tf.summary.histogram('preds', preds)
    labels = tf.placeholder(tf.float32, shape=[None, 1])

    loss = tf.reduce_mean(binary_crossentropy(labels, preds))
    tf.summary.scalar('loss', loss)
    acc_value = tf.reduce_mean(binary_accuracy(labels, preds))
    tf.summary.scalar('accuracy', acc_value)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    merged = tf.summary.merge_all()
    # run model
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('../docs/train', sess.graph)
        test_writer = tf.summary.FileWriter('../docs/test')
        # initializers
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # feed data to training
        number_of_training_data = len(outputs)
        batch_size = 20
        for epoch in range(10):
            for start, end in zip(
                    range(0, number_of_training_data, batch_size),
                    range(batch_size, number_of_training_data, batch_size)):
                # _, trn_loss = sess.run(
                #     [train_step, loss],
                #     feed_dict={
                #         inputs: inputs_1[start:end],
                #         labels: outputs[start:end],
                #         K.learning_phase(): 1
                #     })
                if start % 10 == 0:
                    summary = sess.run(
                        merged,
                        feed_dict={
                            inputs: inputs_1,
                            labels: outputs,
                            K.learning_phase(): 0
                        })
                    test_writer.add_summary(summary, start / 10)
                else:
                    summary, _ = sess.run(
                        [merged, train_step],
                        feed_dict={
                            inputs: inputs_1[start:end],
                            labels: outputs[start:end],
                            K.learning_phase(): 1
                        })
                    train_writer.add_summary(summary, start / 10)
            # print('epoch_step:{}, loss:{}, acc:{}'.format(
            #     epoch, tst_loss, tst_acc))


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [
            layer.output for layer in model.layers if layer.name == layer_name
        ]  # all layer outputs
    funcs = [
        K.function([inp] + [K.learning_phase()], [out]) for out in outputs
    ]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


def get_data(n, input_dim, attention_column=1):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    x = np.random.standard_normal(size=(n, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column] = y[:, 0]
    return x, y


def get_data_recurrent(n, time_steps, input_dim, attention_column=10):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param time_steps: the number of time steps of your series.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    x = np.random.standard_normal(size=(n, time_steps, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column, :] = np.tile(y[:], (1, input_dim))
    return x, y


if __name__ == '__main__':
    N = 10000
    inputs_1, outputs = get_data(N, input_dim)
    build_model()

    # m.fit([inputs_1], outputs, epochs=20, batch_size=64, validation_split=0.5)
    #
    # testing_inputs_1, testing_outputs = get_data(1, input_dim)
    #
    # # Attention vector corresponds to the second matrix.
    # # The first one is the Inputs output.
    # attention_vector = get_activations(m, testing_inputs_1,
    #                                    print_shape_only=True,
    #                                    layer_name='attention_vec')[0].flatten()
    # print('attention =', attention_vector)
    #
    # # plot part.
    # import matplotlib.pyplot as plt
    # import pandas as pd
    #
    # pd.DataFrame(attention_vector, columns=['attention (%)']).plot(kind='bar',
    #                                                                title='Attention Mechanism as '
    #                                                                      'a function of input'
    #                                                                      ' dimensions.')
    # plt.show()

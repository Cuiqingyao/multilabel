# -*- coding:utf-8 -*-
"""
@version: 1.0
@author: kevin
@license: Apache Licence
@contact: liujiezhang@bupt.edu.cn
@site:
@software: PyCharm Community Edition
@file: adios_train.py
@time: 17/05/03 17:39
"""
import json
import os
import time
from math import ceil

import numpy as np
import tensorflow as tf
import yaml
from keras import backend as K
from sklearn import linear_model as lm
from utils.data_helper import build_data_cv
from utils.hiso import HISO
from utils.metrics import (Average_precision, Coverage, Hamming_loss,
                           One_error, Ranking_loss, Construct_thresholds)
K.set_learning_phase(1)


def do_eval(sess, model, thres_model, eval_data, batch_size):
    '''
    eval test data for moedel.
    :param sess:
    :param model:
    :param eval_data:
    :param batch_size:
    :return:
    '''
    K.set_learning_phase(0)
    number_of_data = len(eval_data)
    number_of_batch = ceil(number_of_data / batch_size)
    Y0_labels, Y1_labels, Y0_probs, Y1_probs = [], [], [], []
    eval_loss, eval_cnt = 0., 0.
    for batch in range(number_of_batch):
        start = batch_size * batch
        end = start + min(batch_size, number_of_data - start)

        eval_Y0_labels = [hml.top_label for hml in eval_data[start:end]]
        eval_Y1_labels = [hml.bottom_label for hml in eval_data[start:end]]

        curr_loss, eval_Y0_probs, eval_Y1_probs = sess.run(
            [model.loss, model.Y0_probs, model.Y1_probs],
            feed_dict={
                model.wds: [hml.wds for hml in eval_data[start:end]],
                model.pos: [hml.pos for hml in eval_data[start:end]],
                model.Y0: eval_Y0_labels,
                model.Y1: eval_Y1_labels
                # K.learning_phase(): 0
            })
        eval_loss += curr_loss
        eval_cnt += 1

        Y0_labels.extend(eval_Y0_labels)
        Y1_labels.extend(eval_Y1_labels)
        Y0_probs.extend(eval_Y0_probs)
        Y1_probs.extend(eval_Y1_probs)

    # evaluation metrics
    Y0_labels = np.array(Y0_labels)
    Y1_labels = np.array(Y1_labels)
    Y0_probs = np.array(Y0_probs)
    Y1_probs = np.array(Y1_probs)
    print('\n')
    print('Y0 label:', Y0_labels[3])
    print('Y0 probs:', Y0_probs[3])
    print('Y1 probs:', Y1_probs[3])
    print('Y1 label:', Y1_labels[3])
    print('\n')
    # probs to predict label over thresholds
    # fit_threshold automatally
    Y0_preds = Y0_probs >= 0.7

    # T1 = thres_model.predict(Y1_probs)
    Y1_preds = Y1_probs >= 0.3

    loss_dict = {'eval_loss': eval_loss / eval_cnt, 'Y0': {}, 'Y1': {}}
    # use eval
    func_eval = [
        'Hamming_loss', 'One_error', 'Ranking_loss', 'Coverage',
        'Average_precision'
    ]
    # 0: 伟哥的评判标准， 1：正确的评判标准
    mode = 1
    for func in func_eval:
        if func == 'Hamming_loss':
            loss_dict['Y0'][func] = eval(func)(Y0_labels, Y0_preds, mode=mode)
            loss_dict['Y1'][func] = eval(func)(Y1_labels, Y1_preds, mode=mode)
        else:
            loss_dict['Y0'][func] = eval(func)(Y0_labels, Y0_probs, mode=mode)
            loss_dict['Y1'][func] = eval(func)(Y1_labels, Y1_probs, mode=mode)
    K.set_learning_phase(1)
    return loss_dict


def train(params):
    '''
    训练模型入口
    :param params: 模型参数 dict
    :return:
    '''
    datas, voc, pos, max_length = build_data_cv(
        file_path='../docs/data/HML_JD_ALL.new.dat',
        voc_path='../docs/data/voc.json',
        pos_path='../docs/data/pos.json',
        cv=5)
    # fill params
    params['voc_size'] = len(voc)
    params['pos_size'] = len(pos)
    params['words']['dim'] = max_length
    params['pos']['dim'] = max_length
    print(json.dumps(params, indent=4))

    # split test and train
    test_datas = list(filter(lambda data: data.cv_n == 1, datas))
    train_datas = list(filter(lambda data: data.cv_n != 1, datas))

    print('train dataset: {}'.format(len(train_datas)))
    print('test dataset: {}'.format(len(test_datas)))
    print('max length: {}'.format(max_length))

    # build model
    timestamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    log_test_dir = '../docs/test/%s' % timestamp
    log_train_dir = '../docs/train/%s' % timestamp
    os.mkdir(log_test_dir)
    os.mkdir(log_train_dir)
    # log_baseline_dir = '../docs/baseline/'
    # os.mkdir(log_baseline_dir)

    loss_key = [
        'Hamming_loss', 'One_error', 'Ranking_loss', 'Coverage',
        'Average_precision'
    ]
    # 设置gpu限制
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = params['gpu_fraction']

    number_of_training_data = len(train_datas)
    batch_size = params['batch_size']
    number_of_batch = int(ceil(number_of_training_data / batch_size))
    # 保存最优模型
    model_dir = params['model_dir'] + time.strftime("%Y-%m-%d-%H:%M:%S",
                                                    time.localtime())
    os.mkdir(model_dir)
    model_name = model_dir + '/' + params['model_name']

    with tf.Session(config=config) as sess, tf.device('/gpu:1'):
        hiso = HISO(params)

        saver = tf.train.Saver(max_to_keep=4)
        # baseline_writer = tf.summary.FileWriter(log_baseline_dir)
        test_writer = tf.summary.FileWriter(log_test_dir)
        train_writer = tf.summary.FileWriter(log_train_dir, sess.graph)

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        step = 0
        min_hamming_loss = 10
        best_sess = sess
        rig_labels, rig_probs = [], []
        for epoch in range(params['epoch']):
            # shuffle in each epoch
            train_datas = np.random.permutation(train_datas)

            for batch in range(number_of_batch):
                step += 1
                start = batch_size * batch
                end = start + min(batch_size, number_of_training_data - start)

                wds = [hml.wds for hml in train_datas[start:end]]
                pos = [hml.pos for hml in train_datas[start:end]]
                Y0 = [hml.top_label for hml in train_datas[start:end]]
                Y1 = [hml.bottom_label for hml in train_datas[start:end]]

                trn_loss, trn_Y1_probs, _ = sess.run(
                    [hiso.loss, hiso.Y1_probs, hiso.train_op],
                    feed_dict={
                        hiso.wds: wds,
                        hiso.pos: pos,
                        hiso.Y0: Y0,
                        hiso.Y1: Y1
                        # K.learning_phase(): 1
                    })
                rig_labels.extend(Y1)
                rig_probs.extend(trn_Y1_probs)
                timestamp = time.strftime("%Y-%m-%d-%H:%M:%S",
                                          time.localtime())
                str_loss = '{}:  epoch: {}, step: {},  train_loss: {}'.format(
                    timestamp, epoch, step, trn_loss)
                print(str_loss)

                # log train loss
                if step % params['log_train_every'] == 0:
                    train_writer.add_summary(
                        tf.Summary(value=[
                            tf.Summary.Value(
                                tag="loss", simple_value=trn_loss)
                        ]),
                        step)
                    # base_value = [tf.Summary.Value(tag='Y1_Hamming_loss', simple_value=0.0809),
                    #               tf.Summary.Value(tag='Y1_Average_precision', simple_value=0.4692),
                    #               tf.Summary.Value(tag='Y1_Coverage', simple_value=1.8400),
                    #               tf.Summary.Value(tag='Y1_One_error', simple_value=0.4852),
                    #               tf.Summary.Value(tag='Y1_Hamming_loss', simple_value=0.0809),
                    #               tf.Summary.Value(tag='Y1_Ranking_loss', simple_value=0.2865),
                    #               ]
                    # base_summary = tf.Summary(value=base_value)
                    # baseline_writer.add_summary(base_summary, step)

                # log eval data
                if step % params['log_eval_every'] == 0:
                    # fit Y1 thresholds
                    print('fit thresholds...')
                    T1 = Construct_thresholds(rig_labels, rig_probs)
                    thres_lr = lm.Ridge()
                    thres_lr.fit(rig_probs, T1)
                    print('fit done!')
                    loss_dict = do_eval(sess, hiso, thres_lr, test_datas, batch_size)
                    rig_labels, rig_probs = [], []

                    timestamp = time.strftime("%Y-%m-%d-%H:%M:%S",
                                              time.localtime())
                    str_loss = '{}:  epoch: {}, step: {},  eval_loss: {}'.format(
                        timestamp, epoch, step, loss_dict['eval_loss'])
                    print(str_loss)

                    value = [
                        tf.Summary.Value(
                            tag="loss", simple_value=loss_dict['eval_loss'])
                    ]
                    for key in loss_key:
                        value.append(
                            tf.Summary.Value(
                                tag="Y0_%s" % key,
                                simple_value=loss_dict['Y0'][key]))
                        value.append(
                            tf.Summary.Value(
                                tag="Y1_%s" % key,
                                simple_value=loss_dict['Y1'][key]))
                        if key == 'Hamming_loss':
                            print('Y0_{}:\t{}\tY1_{}:\t{}'.format(
                                key, loss_dict['Y0'][key], key, loss_dict['Y1']
                                [key]))

                    summary = tf.Summary(value=value)
                    test_writer.add_summary(summary, step)
                    # judge whether test_acc is greater than before
                    if loss_dict['Y1']['Hamming_loss'] < min_hamming_loss:
                        min_hamming_loss = loss_dict['Y1']['Hamming_loss']
                        best_sess = sess
                        saver.save(
                            best_sess,
                            model_name + '-%s' % min_hamming_loss,
                            global_step=step,
                            write_meta_graph=True)
                        # predict and save train data
        test_writer.close()
        train_writer.close()
        # predict(
        #     best_sess,
        #     hiso,
        #     datas,
        #     batch_size,
        #     save_name='data-%s.txt' % timestamp)


def predict(sess, model, dataset, batch_size, save_name='eval.csv'):
    '''
    predict labels.
    '''
    print('start to predict labels.....')
    K.set_learning_phase(0)
    number_of_data = len(dataset)
    number_of_batch = int(ceil(number_of_data / batch_size))

    with open('../docs/result/%s' % save_name, 'w') as f:
        for batch in range(number_of_batch):
            print('current process {} -- {}'.format(number_of_batch, batch))
            start = batch_size * batch
            end = start + min(batch_size, number_of_data - start)

            cur_wds = [hml.wds for hml in dataset[start:end]]
            cur_pos = [hml.pos for hml in dataset[start:end]]
            cur_Y0 = [hml.top_label for hml in dataset[start:end]]
            cur_Y1 = [hml.bottom_label for hml in dataset[start:end]]

            curr_Y0_probs, curr_Y1_probs = sess.run(
                [model.Y0_probs, model.Y1_probs],
                feed_dict={
                    model.wds: cur_wds,
                    model.pos: cur_pos
                    # K.learning_phase(): 1
                })

            # transform [1] -> 'POSITIVE'
            for i in range(start, end):
                dataset[i].top_probs = ' '.join(
                    [str(s) for s in curr_Y0_probs[i]])
                dataset[i].bottom_probs = ' '.join(
                    [str(s) for s in curr_Y1_probs[i]])
                line = '{}\t{}\t{}\t{}\t{}\n'.format(' '.join(
                    [str(x)
                     for x in cur_Y1[i]]), dataset[i].bottom_probs, ' '.join(
                         [str(x) for x in cur_Y0[i]]), dataset[i].top_probs,
                                                     dataset[i].raw_sentence)
                f.write(line)

    K.set_learning_phase(1)


def load_predict(model_meta_path,
                 predict_path,
                 save_name='eval.txt',
                 mode='eval',
                 batch_size=128):
    '''
    load lastest model and predict datas.
    :return:
    '''
    # 动态申请gpu，用多少申请多少
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph(model_meta_path)
        saver.restore(
            sess, tf.train.latest_checkpoint(os.path.dirname(model_meta_path)))

        # load graph
        graph = tf.get_default_graph()

        # get input placeholder
        tf_wds = graph.get_tensor_by_name('words:0')
        tf_pos = graph.get_tensor_by_name('pos:0')
        # tf_combine_feature = graph.get_tensor_by_name('combine_feature:0')

        tf_Y0_probs = graph.get_tensor_by_name('Y0_probs:0')
        tf_Y1_probs = graph.get_tensor_by_name('Y1_probs:0')

        model = TFModel(tf_wds, tf_pos, tf_Y0_probs, tf_Y1_probs)

        # predict and save eval data
        datas, vocab, pos, max_length = build_data_cv(
            file_path=predict_path,
            voc_path='../docs/voc.json',
            pos_path='../docs/pos.json',
            cv=5)
        predict(sess, model, datas, batch_size, save_name=save_name)


class TFModel(object):
    def __init__(self, wds, pos, Y0_probs, Y1_probs):
        self.wds = wds
        self.pos = pos
        self.Y0_probs = Y0_probs
        self.Y1_probs = Y1_probs


if __name__ == '__main__':
    # load params
    params = yaml.load(open('./utils/params.yaml', 'r'))

    train(params)

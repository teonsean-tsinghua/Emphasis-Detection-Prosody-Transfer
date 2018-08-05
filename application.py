'''
已有：中文文本，英文文本，英文语音，英文语音特征，相关度矩阵
1. 对带有韵律对中文语音进行SPPAS切分对齐
2. 提取特征并计算中文diff
3. 输入到网络中，得到英文diff
4. 计算新时长
5. 计算新f0, e
'''

import os
import re
import subprocess
import xml.etree.ElementTree as ET
from speechFeatures import features_util as fu
import numpy as np
import math
from sklearn import preprocessing
import pickle as pk
import tensorflow as tf
from tensorflow.contrib import rnn, layers
from tensorflow import nn

path = os.popen('pwd').readlines()[0].strip()


def translate():
    global align
    tree = ET.parse('./tmp/palign.xra')
    doc = tree.getroot()
    active_time = []
    alignment = []
    tokens = []
    for child in doc:
        if child.tag != 'Tier':
            continue
        if child.attrib['tiername'] == 'Activity':
            for annotation in child:
                if annotation.find('Label').find('Tag').text == 'speech':
                    interval = annotation.find('Location').find('Interval')
                    begin = int(float(interval.find('Begin').get('midpoint')) * 1000)
                    end = int(float(interval.find('End').get('midpoint')) * 1000)
                    active_time.append((begin, end))
        elif child.attrib['tiername'] == 'TokensAlign':
            for annotation in child:
                tag = annotation.find('Label').find('Tag').text
                if tag == '#':
                    continue
                interval = annotation.find('Location').find('Interval')
                begin = int(float(interval.find('Begin').get('midpoint')) * 1000)
                end = int(float(interval.find('End').get('midpoint')) * 1000)
                alignment.append((tag, begin, end))
                tokens.append(tag)
    return active_time, alignment, tokens


def parallelize(text, filename, language):
    commands = []
    commands.append('cd %s' % path)
    commands.append('echo %s | ' % text +
                    'python ./SPPAS/sppas/bin/normalize.py ' +
                    '-r ./SPPAS/resources/vocab/%s.vocab ' % ('cmn' if language == 'cn' else 'eng') +
                    '> ./tmp/text.txt')
    commands.append('python ./SPPAS/sppas/bin/wavsplit.py ' +
                    '-w ./test/%s ' % filename +
                    '-t ./tmp/text.txt ' +
                    '-p ./tmp/segm.xra')
    commands.append('python ./SPPAS/sppas/bin/normalize.py ' +
                    '-i ./tmp/segm.xra ' +
                    '-r ./SPPAS/resources/vocab/%s.vocab ' % ('cmn' if language == 'cn' else 'eng') +
                    '-o ./tmp/norm.xra')
    commands.append('python ./SPPAS/sppas/bin/phonetize.py ' +
                    '-i ./tmp/norm.xra ' +
                    '-r ./SPPAS/resources/dict/%s.dict ' % ('cmn' if language == 'cn' else 'eng') +
                    '-o ./tmp/phon.xra')
    commands.append('python ./SPPAS/sppas/bin/alignment.py ' +
                    '-w ./test/%s ' % filename +
                    '-i ./tmp/phon.xra ' +
                    '-I ./tmp/norm.xra ' +
                    '-r ./SPPAS/resources/models/models-%s ' % ('cmn' if language == 'cn' else 'eng') +
                    '-o ./tmp/palign.xra')
    subprocess.call(' && '.join(commands), shell=True)
    return translate()


def count(s):
    total_syllables = 0
    s = re.sub(r'qu', 'qw', s)
    s = re.sub(r'(es$)|(que$)|(gue$)', '', s)
    s = re.sub(r'^re', r'ren', s)
    s = re.sub(r'^gua', r'ga', s)
    s = re.sub(r'([aeiou])(l+e$)', r'\g<1>', s)
    (s, syllables) = re.subn(r'([bcdfghjklmnpqrstvwxyz])(l+e$)', r'\g<1>', s)
    total_syllables += syllables
    s = re.sub(r'([aeiou])(ed$)', r'\g<1>', s)
    (s, syllables) = re.subn(r'([bcdfghjklmnpqrstvwxyz])(ed$)', r'\g<1>', s)
    total_syllables += syllables
    endsp = re.compile(r'(ly$)|(ful$)|(ness$)|(ing$)|(est$)|(er$)|(ent$)|(ence$)')
    (s, syllables) = endsp.subn(r'', s)
    total_syllables += syllables
    (s, syllables) = endsp.subn(r'', s)
    total_syllables += syllables
    s = re.sub(r'(^y)([aeiou][aeiou]*)', r'\g<2>', s)
    s = re.sub(r'([aeiou])(y)', r'\g<1>t', s)
    s = re.sub(r'aa+', r'a', s)
    s = re.sub(r'ee+', r'e', s)
    s = re.sub(r'ii+', r'i', s)
    s = re.sub(r'oo+', r'o', s)
    s = re.sub(r'uu+', r'u', s)
    dipthongs = re.compile(r'(eau)|(iou)|(are)|(ai)|(au)|(ea)|(ei)|(eu)|(ie)|(io)|(oa)|(oe)|(oi)|(ou)|(ue)|(ui)')
    s, syllables = dipthongs.subn('', s)
    total_syllables += syllables
    if len(s) > 3:
        s = re.sub(r'([bcdfghjklmnpqrstvwxyz])(e$)', r'\g<1>', s)
    s, syllables = re.subn(r'[aeiouy]', '', s)
    total_syllables += syllables
    if total_syllables == 0:
        total_syllables = 1
    return total_syllables


def extraction(filename, at, al, language):
    def time_to_frame(start, end):
        return [start // 5, end // 5 - 1]

    wav_info = fu.read_wav('test/%s' % filename)
    frames = fu.enframe(wav_info['signal'], wav_info['fs'], 10, 5)
    active_time = at
    active_range = list(map(lambda interval: time_to_frame(interval[0], interval[1]), active_time))
    sentence_unit = al
    units = list(map(lambda interval: [interval[0]] + time_to_frame(interval[1], interval[2]), sentence_unit))

    '''E'''
    E = fu.short_time_energy(frames)
    raw_E = np.array(list(map(lambda unit: np.average(E[unit[1]:unit[2]]), units))).reshape(-1, 1)
    E_scaler = preprocessing.MinMaxScaler()
    norm_E = E_scaler.fit_transform(raw_E)

    '''F0'''
    avg_F0 = []
    for interval in active_range:
        avg_F0 = np.hstack(avg_F0 + [fu.basic_frequency(wav_info['signal'][i * 80:(i * 80 + 320)],
                                                        wav_info['fs']) for i in range(interval[0], interval[1] - 2)])
    raw_F0 = []
    for unit in units:
        raw_F0.append(np.average([fu.basic_frequency(wav_info['signal'][i * 80:(i * 80 + 320)],
                                                     wav_info['fs']) for i in range(unit[1], unit[2] - 2)]))
    raw_F0 = np.array(raw_F0).reshape(-1, 1)
    F0_scaler = preprocessing.MinMaxScaler()
    norm_F0 = F0_scaler.fit_transform(raw_F0)

    '''duration'''
    raw_duration = []
    for unit in sentence_unit:
        if language == 'cn':
            cnt = len(unit[0])
        else:
            cnt = count(unit[0])
        raw_duration.append((unit[2] - unit[1]) / cnt)
    raw_duration = np.array(raw_duration).reshape(-1, 1)
    Duration_scaler = preprocessing.MinMaxScaler()
    norm_duration = Duration_scaler.fit_transform(raw_duration)

    with open('test/%s.feat' % filename, 'wb') as out:
        pk.dump((norm_E, E_scaler, norm_F0, F0_scaler, norm_duration, Duration_scaler), out)


def correlations(cn_words, en_words, align):
    alignment = np.array([[0] * len(cn_words)] * len(en_words))
    matches = align.split(', ')
    for match in matches:
        nums = match.split('-')
        alignment[int(nums[1])][int(nums[0])] = 1
    correlation = np.array([[0] * len(cn_words)] * len(en_words), dtype=float)
    print(alignment)
    softmax = []
    for i in range(correlation.shape[0]):
        for j in range(correlation.shape[1]):
            correlation[i][j] = 0
            for k in range(correlation.shape[1]):
                correlation[i][j] += alignment[i][k] * (-math.exp(math.fabs(j - k)))
        softmax.append(list(np.exp(correlation[i]) / np.sum(np.exp(correlation[i]))))
    print(np.array(softmax))
    return softmax


# et = open('test/en_text.txt', 'r').readlines()
#
# with open('test/en.talign', 'w') as out:
#     for i, t in enumerate(et):
#         re = parallelize(t.strip(), 'en_%d.wav' % i, 'en')
#         out.write(str(re) + '\n')

# ct = open('test/cn_text.txt', 'r').readlines()
#
# with open('test/cn.talign', 'w') as out:
#     for i, t in enumerate(ct):
#         re = parallelize(t.strip(), 'cn_%d.wav' % i, 'cn')
#         out.write(str(re) + '\n')

# ca = open('test/cn.talign', 'r').readlines()
# ea = open('test/en.talign', 'r').readlines()
# align = open('test/align.txt', 'r').readlines()
# with open('test/correlation.txt', 'w') as out:
#     for ctokens, etokens, ali in zip(ca, ea, align):
#         out.write(str(correlations(eval(ctokens.strip())[2],
#                                    eval(etokens.strip())[2],
#                                    ali.strip())) + '\n')
#
# et = open('test/en_text.txt', 'r').readlines()
# ct = open('test/cn_text.txt', 'r').readlines()
# ea = open('test/en.talign', 'r').readlines()
# corr = open('test/correlation.txt', 'r').readlines()
# ea = {0: eval(ea[0].strip()), 1: eval(ea[1].strip())}
# ct = {0: ct[0].strip(), 1: ct[1].strip()}
# et = {0: et[0].strip(), 1: et[1].strip()}
# corr = {0: eval(corr[0].strip()), 1: eval(corr[1].strip())}

# for sentence in range(2):
#     for idx in range(4):
#         filename = '%d_%d.wav' % (sentence, idx)
#         active_time_, alignment_, tokens_ = parallelize(ct[sentence], filename, 'cn')
#         extraction(filename, active_time_, alignment_, 'cn')
#     filename = 'en_%d.wav' % sentence
#     active_time_, alignment_, tokens_ = parallelize(et[sentence], filename, 'en')
#     extraction(filename, active_time_, alignment_, 'en')

# learning_rate = 0.001
#
# training_iters = 20000
#
# batch_size = 1
#
# display_step = 50
#
# n_input = 3
#
# n_bilstm_hidden = 128
#
# n_bilstm_output = 128
#
# n_bilstm2_hidden = 128
#
# n_bilstm2_output = 3
#
# regularization_scale = 0.001
#
# cn = tf.placeholder("float", [1, None, n_input])
# cor = tf.placeholder("float", [None, None])
#
# weights = {
#     'bilstm': tf.Variable(tf.random_normal([2*n_bilstm_hidden, n_bilstm_output])),
#     'bilstm2': tf.Variable(tf.random_normal([2*n_bilstm2_hidden, n_bilstm2_output]))
# }
#
# biases = {
#     'bilstm': tf.Variable(tf.random_normal([n_bilstm_output])),
#     'bilstm2': tf.Variable(tf.random_normal([n_bilstm2_output]))
# }
#
#
# def BiLSTM_Correlation_BiLSTM(cn, cor, weights, biases):
#     # BiLSTM
#     lstm_fw_cell = rnn.BasicLSTMCell(n_bilstm_hidden, forget_bias=1.0)
#     lstm_bw_cell = rnn.BasicLSTMCell(n_bilstm_hidden, forget_bias=1.0)
#     bilstm_outputs, _ = nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, cn,
#                                                      dtype=tf.float32, scope='input')
#     bilstm_output = tf.matmul(tf.concat(bilstm_outputs, axis=2)[0], weights['bilstm']) + biases['bilstm']
#
#     # Attention
#     lstm2_input = tf.convert_to_tensor([tf.matmul(cor, bilstm_output)])
#
#     # BiLSTM
#     lstm2_fw_cell = rnn.BasicLSTMCell(n_bilstm2_hidden, forget_bias=1.0)
#     lstm2_bw_cell = rnn.BasicLSTMCell(n_bilstm2_hidden, forget_bias=1.0)
#     bilstm2_outputs, _ = nn.bidirectional_dynamic_rnn(lstm2_fw_cell, lstm2_bw_cell, lstm2_input,
#                                                       dtype=tf.float32, scope='output')
#     bilstm2_output = tf.matmul(tf.concat(bilstm2_outputs, axis=2)[0], weights['bilstm2']) + biases['bilstm2']
#
#     return bilstm2_output
#
#
# estimation = BiLSTM_Correlation_BiLSTM(cn, cor, weights, biases)
#
# init = tf.global_variables_initializer()
#
# saver = tf.train.Saver(max_to_keep=5)
#
# for modelidx in [3999, 7999, 11999, 15999, 19999]:
#     with tf.Session() as sess:
#         sess.run(init)
#         saver.restore(sess, 'ckpt/BCB.ckpt-%d' % modelidx)
#         for sentence in range(2):
#             for idx in range(4):
#                 filename = '%d_%d.wav' % (sentence, idx)
#                 with open('test/%s.feat' % filename, 'rb') as f:
#                     norm_E, E_scaler, norm_F0, F0_scaler, norm_duration, Duration_scaler = pk.load(f)
#                 cn_f = np.concatenate([norm_E, norm_F0, norm_duration], axis=1)
#                 output = sess.run(estimation, feed_dict={cn: [cn_f], cor: corr[sentence]})
#                 with open('test/m%d_%d_%d.out' % (modelidx, sentence, idx), 'w') as f:
#                     f.write(str(list(map(lambda x: list(x), output))))
#
# def read_float32(filename, num):
#    abc = np.fromfile(filename, dtype='float32')
#    nlen = int(len(abc) / num)
#    info = abc.reshape(nlen, num)
#    return info
#
#
# def write_float32(filename, info):
#     info.tofile(filename)
#
#
# d = {0: {0: [1], 1: [2], 2: [4], 3: [5, 6, 7], 4: [8, 9, 10, 11, 12], 5: [13, 14, 15], 6: [16, 17, 18, 19, 20]},
#      1: {0: [i for i in range(1, 13)], 1: [14], 2: [15, 16, 17], 3: [18, 19, 20, 21], 4: [22, 23],
#          5: [i for i in range(24, 33)], 6: [33, 34, 35, 36], 7: [37, 38, 39, 40], 8: [41, 42, 43, 44, 45]}}
# bound = [21, 46]
#
#
alphas = [0.2, 0.5, 0.8]


# for sentence in range(2):
#     for idx in range(4):
#         info = read_float32('test/%d.dat' % sentence, 6)
#         duration = np.array(list(map(lambda x: x[0], info)))
#         selected_duration = []
#         selected_duration_dict = {}
#         for k, v in d[sentence].items():
#             for vv in v:
#                 selected_duration.append(duration[vv])
#                 selected_duration_dict[vv] = len(selected_duration) - 1
#         selected_duration = np.array(selected_duration)
#         norm = preprocessing.MinMaxScaler()
#         selected_duration = norm.fit_transform(selected_duration.reshape(-1, 1))
#         selected_duration = selected_duration.reshape(1, -1)[0]
#         for modelidx in [3999, 7999, 11999, 15999, 19999]:
#             feat = eval(open('test/m%d_%d_%d.out' % (modelidx, sentence, idx), 'r').readlines()[0].strip())
#             for alpha in alphas:
#                 modified_selected_duration = [i for i in selected_duration]
#                 modified_info = np.copy(info)
#                 for i, norm_ in enumerate(feat):
#                     for j in d[sentence][i]:
#                         k = selected_duration_dict[j]
#                         modified_selected_duration[k] = alpha * modified_selected_duration[k] + (1 - alpha) * norm_[2]
#                         modified_info[j][0] = norm.inverse_transform(modified_selected_duration[k])
#                         multi = modified_info[j][0] / info[j][0]
#                         for t in range(1, 6):
#                             modified_info[j][t] *= multi
#                 write_float32('test/time/m%d_%d_%d_%d.dat' % (modelidx, sentence, idx, int(alpha * 10)),
#                               np.concatenate((modified_info, [[0, 0, 0, 0, 0, 1]] * len(modified_info)),
#                                              axis=1).astype(np.float32))

for sentence in range(2):
    for idx in range(4):
        for midx in [3999, 7999, 11999, 15999, 19999]:
            feat = eval(open('test/m%d_%d_%d.out' % (midx, sentence, idx), 'r').readlines()[0].strip())
            for alpha in alphas:
                filename = 'm%d_%d_%d_%d.npy' % (midx, sentence, idx, int(alpha * 10))
                boundary = np.load('test/split/%s' % filename)
                features = np.load('test/features/%s' % filename)
                selected_E = []
                selected_F0 = []
                for bnd in boundary:
                    if bnd[2] == 'pau':
                        continue
                    start = int(bnd[0].strip())
                    end = int(bnd[1].strip())
                    for frame in range(start, end + 1):
                        selected_F0.append(features[frame][1])
                        selected_E.append(features[frame][7])
                E_scaler = preprocessing.MinMaxScaler()
                F_scaler = preprocessing.MinMaxScaler()
                selected_E = np.array(selected_E)
                selected_F0 = np.array(selected_F0)
                selected_E = E_scaler.fit(selected_E.reshape(-1, 1))
                selected_F0 = F_scaler.fit(selected_F0.reshape(-1, 1))
                step = 0
                for bnd in boundary:
                    if bnd[2] == 'pau':
                        continue
                    start = int(bnd[0].strip())
                    end = int(bnd[1].strip())
                    for frame in range(start, end + 1):
                        F0 = F_scaler.transform(features[frame][1])
                        F0 = F0 * alpha + (1 - alpha) * feat[step][1]
                        features[frame][1] = F_scaler.inverse_transform(F0)
                        E = E_scaler.transform(features[frame][7])
                        E = E * alpha + (1 - alpha) * feat[step][0]
                        features[frame][7] = E_scaler.inverse_transform(E)
                    step += 1
                np.save('test/new/m%d_%d_%d_%d.npy' % (midx, sentence, idx, int(alpha * 10)),
                        features.astype(np.float32))

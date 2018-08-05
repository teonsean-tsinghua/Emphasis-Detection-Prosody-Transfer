import tensorflow as tf
from tensorflow.contrib import rnn, layers
from tensorflow import nn
import numpy as np
import pandas as pd
import random

def next_pair(mode):
    meta = pd.read_csv('csv/dataset_meta.csv')
    features = pd.read_csv('csv/dataset_feature.csv')
    meta = meta[meta['Token correlation'].notna()]
    male_meta = meta[meta['Gender'] == 'male']
    female_meta = meta[meta['Gender'] == 'female']
    cn_male_feature = features[(features['Language'] == 'cn') & (features['Gender'] == 'male')]
    cn_female_feature = features[(features['Language'] == 'cn') & (features['Gender'] == 'female')]
    en_male_feature = features[(features['Language'] == 'en') & (features['Gender'] == 'male')]
    en_female_feature = features[(features['Language'] == 'en') & (features['Gender'] == 'female')]

    def get_feature_vec(language, gender, filename):
        if language == 'cn':
            if gender == 'male':
                df = cn_male_feature[cn_male_feature['Filename'] == filename]
            elif gender == 'female':
                df = cn_female_feature[cn_female_feature['Filename'] == filename]
        elif language == 'en':
            if gender == 'male':
                df = en_male_feature[en_male_feature['Filename'] == filename]
            elif gender == 'female':
                df = en_female_feature[en_female_feature['Filename'] == filename]
        return np.array(df[['Norm E', 'Norm F0', 'Norm duration']])

    while True:
        if mode == 'male':
            idx = random.randint(0, len(male_meta) - 1)
            cnfile = male_meta.iloc[idx]['CnFile']
            enfile = male_meta.iloc[idx]['EnFile']
            yield get_feature_vec('cn', 'male', cnfile),\
                  get_feature_vec('en', 'male', enfile),\
                  np.array(eval(male_meta.iloc[idx]['Token correlation']))
        elif mode == 'female':
            idx = random.randint(0, len(female_meta) - 1)
            cnfile = female_meta.iloc[idx]['CnFile']
            enfile = female_meta.iloc[idx]['EnFile']
            yield get_feature_vec('cn', 'female', cnfile),\
                  get_feature_vec('en', 'female', enfile),\
                  np.array(eval(female_meta.iloc[idx]['Token correlation']))
        elif mode == 'both':
            idx = random.randint(0, len(meta) - 1)
            cnfile = meta.iloc[idx]['CnFile']
            enfile = meta.iloc[idx]['EnFile']
            gender = meta.iloc[idx]['Gender']
            yield get_feature_vec('cn', gender, cnfile),\
                  get_feature_vec('en', gender, enfile),\
                  np.array(eval(meta.iloc[idx]['Token correlation']))


learning_rate = 0.001

training_iters = 20000

batch_size = 1

display_step = 50

n_input = 3

n_bilstm_hidden = 128

n_bilstm_output = 128

n_bilstm2_hidden = 128

n_bilstm2_output = 3

regularization_scale = 0.001

cn = tf.placeholder("float", [1, None, n_input])
en = tf.placeholder("float", [None, n_bilstm2_output])
cor = tf.placeholder("float", [None, None])

weights = {
    'bilstm': tf.Variable(tf.random_normal([2*n_bilstm_hidden, n_bilstm_output])),
    'bilstm2': tf.Variable(tf.random_normal([2*n_bilstm2_hidden, n_bilstm2_output]))
}

biases = {
    'bilstm': tf.Variable(tf.random_normal([n_bilstm_output])),
    'bilstm2': tf.Variable(tf.random_normal([n_bilstm2_output]))
}


def BiLSTM_Correlation_BiLSTM(cn, cor, weights, biases):
    # BiLSTM
    lstm_fw_cell = rnn.BasicLSTMCell(n_bilstm_hidden, forget_bias=1.0)
    lstm_bw_cell = rnn.BasicLSTMCell(n_bilstm_hidden, forget_bias=1.0)
    bilstm_outputs, _ = nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, cn,
                                                     dtype=tf.float32, scope='input')
    bilstm_output = tf.matmul(tf.concat(bilstm_outputs, axis=2)[0], weights['bilstm']) + biases['bilstm']

    # Attention
    lstm2_input = tf.convert_to_tensor([tf.matmul(cor, bilstm_output)])

    # BiLSTM
    lstm2_fw_cell = rnn.BasicLSTMCell(n_bilstm2_hidden, forget_bias=1.0)
    lstm2_bw_cell = rnn.BasicLSTMCell(n_bilstm2_hidden, forget_bias=1.0)
    bilstm2_outputs, _ = nn.bidirectional_dynamic_rnn(lstm2_fw_cell, lstm2_bw_cell, lstm2_input,
                                                      dtype=tf.float32, scope='output')
    bilstm2_output = tf.matmul(tf.concat(bilstm2_outputs, axis=2)[0], weights['bilstm2']) + biases['bilstm2']

    return bilstm2_output


estimation = BiLSTM_Correlation_BiLSTM(cn, cor, weights, biases)

loss = tf.reduce_mean(tf.square(en - estimation)) + \
       tf.contrib.layers.apply_regularization(layers.l1_regularizer(regularization_scale),
                                              tf.trainable_variables())
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    sess.run(init)
    step = 1
    gen = next_pair('male')
    while step * batch_size < training_iters:
        cn_f, en_f, correlation = next(gen)
        sess.run(optimizer, feed_dict={cn: [cn_f], en: en_f, cor: correlation})
        if step % display_step == 0:
            train_loss = sess.run(loss, feed_dict={en: en_f, cor: correlation, cn: [cn_f]})
            print("Iter " + str(step*batch_size) +
                  ", Minibatch Loss= " +
                  "{:.6f}".format(train_loss))
        if (step + 1) % 4000 == 0:
            saver.save(sess, 'ckpt/BCB.ckpt', global_step=step)
        step += 1
    print("Optimization Finished!")

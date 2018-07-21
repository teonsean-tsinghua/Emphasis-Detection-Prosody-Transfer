import pandas as pd
from speechFeatures import features_util as fu
import numpy as np
from sklearn import preprocessing
import re
import sys


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
    return total_syllables


time_align = pd.read_csv('csv/dataset_time_align.csv', header=0)
feature_avg = pd.DataFrame(columns=('Language', 'Gender', 'Filename', 'Avg E', 'Avg F0', 'Avg MFCC', 'Avg duration'))
feature = pd.DataFrame(columns=('Language', 'Gender', 'Filename', 'Text', 'Start', 'End',
                                'Raw E', 'Norm E', 'Diff E', 'Z0 E','NormZ0 E',
                                'Raw F0', 'Norm F0', 'Diff F0', 'Z0 F0', 'NormZ0 F0',
                                'Raw duration', 'Norm duration', 'Diff duration', 'NormZ0 duration', 'Z0 duration',
                                'NormZ0 MFCC', 'Diff MFCC'))
for i, row in time_align.iterrows():
    def time_to_frame(start, end):
        return [start // 5, end // 5 - 1]

    sys.stdout.write('processing feature extraction: %d/%d\r' % (i + 1, len(time_align)))
    sys.stdout.flush()
    wav_info = fu.read_wav('audio/segment/%s/%s/%s' % (row['Language'], row['Gender'], row['Filename']))
    fa_item = {'Language': row['Language'], 'Gender': row['Gender'], 'Filename': row['Filename']}
    frames = fu.enframe(wav_info['signal'], wav_info['fs'], 10, 5)
    active_time = eval(row['Active time'])
    active_range = list(map(lambda interval: time_to_frame(interval[0], interval[1]), active_time))
    sentence_unit = eval(row['Alignment'])
    units = list(map(lambda interval: [interval[0]] + time_to_frame(interval[1], interval[2]), sentence_unit))

    '''E'''
    E = fu.short_time_energy(frames)
    fa_item['Avg E'] = np.average(np.hstack(map(lambda interval: E[interval[0]:interval[1]], active_range)))
    raw_E = np.array(list(map(lambda unit: np.average(E[unit[1]:unit[2]]), units))).reshape(-1, 1)
    E_scaler = preprocessing.MinMaxScaler()
    norm_E = E_scaler.fit_transform(raw_E)
    norm_avg_E = E_scaler.transform(fa_item['Avg E'])
    diff_E = (norm_E - norm_avg_E) / norm_avg_E
    z0_E = preprocessing.StandardScaler().fit_transform(raw_E)
    normz0_E = preprocessing.StandardScaler().fit_transform(norm_E)
    '''F0'''
    avg_F0 = []
    for interval in active_range:
        avg_F0 = np.hstack(avg_F0 + [fu.basic_frequency(wav_info['signal'][i * 80:(i * 80 + 320)],
                                                        wav_info['fs']) for i in range(interval[0], interval[1] - 2)])
    fa_item['Avg F0'] = np.average(avg_F0)
    raw_F0 = []
    for unit in units:
        raw_F0.append(np.average([fu.basic_frequency(wav_info['signal'][i * 80:(i * 80 + 320)],
                                                     wav_info['fs']) for i in range(unit[1], unit[2] - 2)]))
    raw_F0 = np.array(raw_F0).reshape(-1, 1)
    F0_scaler = preprocessing.MinMaxScaler()
    norm_F0 = F0_scaler.fit_transform(raw_F0)
    norm_avg_F0 = F0_scaler.transform(fa_item['Avg F0'])
    diff_F0 = (norm_F0 - norm_avg_F0) / norm_avg_F0
    z0_F0 = preprocessing.StandardScaler().fit_transform(raw_F0)
    normz0_F0 = preprocessing.StandardScaler().fit_transform(norm_F0)

    '''duration'''
    syllable = 0
    raw_duration = []
    for unit in sentence_unit:
        if row['Language'] == 'cn':
            cnt = len(unit[0])
        else:
            cnt = count(unit[0])
        raw_duration.append((unit[2] - unit[1]) / cnt)
        syllable += cnt
    fa_item['Avg duration'] = np.sum(list(map(lambda interval: interval[1] - interval[0],
                                              active_time))) / syllable
    raw_duration = np.array(raw_duration).reshape(-1, 1)
    Duration_scaler = preprocessing.MinMaxScaler()
    norm_duration = Duration_scaler.fit_transform(raw_duration)
    norm_avg_duration = Duration_scaler.transform(fa_item['Avg duration'])
    diff_duration = (norm_duration - norm_avg_duration) / norm_avg_duration
    z0_duration = preprocessing.StandardScaler().fit_transform(raw_duration)
    normz0_duration = preprocessing.StandardScaler().fit_transform(norm_duration)

    '''MFCC'''
    mfccs = []
    for interval in active_time:
        mfccs.append(fu.mfcc(wav_info['signal'][interval[0] * 16:interval[1] * 16],
                             wav_info['fs']))
    mfccs = np.hstack(mfccs)
    mfcc_scalers = []
    mfcc_normalizers = []
    for i, dimension in enumerate(mfccs):
        normalizer = preprocessing.MinMaxScaler()
        scaler = preprocessing.StandardScaler()
        mfccs[i] = normalizer.fit_transform(scaler.fit_transform(dimension.reshape(-1, 1))).reshape(1, -1)
        mfcc_scalers.append(scaler)
        mfcc_normalizers.append(normalizer)
    avg_mfcc = np.array([np.average(column) for column in mfccs])
    fa_item['Avg MFCC'] = avg_mfcc
    normz0_mfcc = []
    for interval in sentence_unit:
        re_mfcc = fu.mfcc(wav_info['signal'][interval[1] * 16:interval[2] * 16], wav_info['fs'])
        for i, dimension in enumerate(re_mfcc):
            re_mfcc[i] = normalizer.transform(scaler.transform(dimension.reshape(-1, 1))).reshape(1, -1)
        normz0_mfcc.append([np.average(column) for column in re_mfcc])
    normz0_mfcc = np.array(normz0_mfcc)
    diff_mfcc = (np.array(normz0_mfcc) - fa_item['Avg MFCC']) / fa_item['Avg MFCC']

    '''merge'''
    feature_avg = feature_avg.append(pd.Series(fa_item), ignore_index=True)
    for i, unit in enumerate(sentence_unit):
        d = {'Language': row['Language'], 'Gender': row['Gender'], 'Filename': row['Filename']}
        d.update({'Text': unit[0], 'Start': unit[1], 'End': unit[2]})
        d.update({'Raw E': raw_E[i][0], 'Raw F0': raw_F0[i][0], 'Raw duration': raw_duration[i][0]})
        d.update({'Norm E': norm_E[i][0], 'Norm F0': norm_F0[i][0], 'Norm duration': norm_duration[i][0]})
        d.update({'Diff E': diff_E[i][0], 'Diff F0': diff_F0[i][0], 'Diff duration': diff_duration[i][0]})
        d.update({'Z0 E': z0_E[i][0], 'Z0 F0': z0_F0[i][0], 'Z0 duration': z0_duration[i][0]})
        d.update({'NormZ0 E': normz0_E[i][0], 'NormZ0 F0': normz0_F0[i][0], 'NormZ0 duration': normz0_duration[i][0]})
        d.update({'Diff MFCC': diff_mfcc[i], 'NormZ0 MFCC': normz0_mfcc[i]})
        feature = feature.append(pd.Series(d), ignore_index=True)
print('\nSaving data to csv files...')
feature_avg.to_csv('csv/dataset_feature_avg.csv', index=False)
feature.to_csv('csv/dataset_feature.csv', index=False)
print('Done')

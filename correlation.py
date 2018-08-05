import pandas as pd
import numpy as np
import math

meta = pd.read_csv('csv/dataset_meta.csv', header=0)
time_align = pd.read_csv('csv/dataset_time_align.csv', header=0)
word_align = list(map(lambda x: x.strip(), open('csv/align.txt', 'r').readlines()))
correlations = []

for idx, row in meta.iterrows():
    if word_align[idx] == 'null':
        correlations.append(None)
        continue
    cn_words = time_align[(time_align['Language'] == 'cn') &
                          (time_align['Gender'] == row['Gender']) &
                          (time_align['Filename'] == row['CnFile'])].iloc[0]['Tokens'].split(' ')
    en_words = time_align[(time_align['Language'] == 'en') &
                          (time_align['Gender'] == row['Gender']) &
                          (time_align['Filename'] == row['EnFile'])].iloc[0]['Tokens'].split(' ')
    alignment = np.array([[0] * len(cn_words)] * len(en_words))
    matches = word_align[idx].split(', ')
    for match in matches:
        nums = match.split('-')
        alignment[int(nums[1])][int(nums[0])] = 1
    correlation = np.array([[0] * len(cn_words)] * len(en_words), dtype=float)
    softmax = []
    for i in range(correlation.shape[0]):
        for j in range(correlation.shape[1]):
            correlation[i][j] = 0
            for k in range(correlation.shape[1]):
                correlation[i][j] += alignment[i][k] * (-math.exp(math.fabs(j - k)))
        softmax.append(list(np.exp(correlation[i]) / np.sum(np.exp(correlation[i]))))
    correlations.append(softmax)
meta.pop('Token correlation')
meta.insert(5, 'Token correlation', pd.Series(correlations))
meta.to_csv('csv/dataset_meta.csv', index=False)
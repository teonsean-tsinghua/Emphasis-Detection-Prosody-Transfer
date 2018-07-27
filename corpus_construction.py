import pandas as pd
import os


align = pd.read_csv('csv/dataset_time_align.csv', header=0)
meta = pd.read_csv('csv/dataset_meta.csv', header=0)

cn_male = align[align['Gender'].isin(['male']) & align['Language'].isin(['cn'])]
cn_female = align[align['Gender'].isin(['female']) & align['Language'].isin(['cn'])]
en_male = align[align['Gender'].isin(['male']) & align['Language'].isin(['en'])]
en_female = align[align['Gender'].isin(['female']) & align['Language'].isin(['en'])]


def print_alignment(walign):
    for _, row in walign.iterrows():
        if row['Gender'] == 'male':
            cn_tokens = cn_male[cn_male['Filename'].isin([row['CnFile']])].iloc[0]['Tokens'].split(' ')
            en_tokens = en_male[en_male['Filename'].isin([row['EnFile']])].iloc[0]['Tokens'].split(' ')
        else:
            cn_tokens = cn_female[cn_female['Filename'].isin([row['CnFile']])].iloc[0]['Tokens'].split(' ')
            en_tokens = en_female[en_female['Filename'].isin([row['EnFile']])].iloc[0]['Tokens'].split(' ')
        aligns = row['Alignment'].split(' ')
        for pair in aligns:
            idxes = pair.split('-')
            print(cn_tokens[int(idxes[0])] + '-' + en_tokens[int(idxes[1])])


pairs = []
pair_info = []
for _, row in meta.iterrows():
    if row['Gender'] == 'male':
        cntokens = cn_male[cn_male['Filename'].isin([row['CnFile']])].iloc[0]['Tokens']
        entokens = en_male[en_male['Filename'].isin([row['EnFile']])].iloc[0]['Tokens']
    else:
        cntokens = cn_female[cn_female['Filename'].isin([row['CnFile']])].iloc[0]['Tokens']
        entokens = en_female[en_female['Filename'].isin([row['EnFile']])].iloc[0]['Tokens']
    pairs.append(cntokens + ' ||| ' + entokens + '\n')
    pair_info.append({'Gender': row['Gender'], 'CnFile': row['CnFile'], 'EnFile': row['EnFile']})
corpus = pd.read_csv('csv/bilingual.csv', header=0)
for _, row in corpus.iterrows():
    pairs.append(row['cn'] + ' ||| ' + row['en'] + '\n')
try:
    os.mkdir('tmp')
except FileExistsError:
    pass
try:
    with open('tmp/corpus.txt', 'w') as fout:
        fout.writelines(pairs)
except BaseException:
    pass
os.system('./fast_align -i tmp/corpus.txt -d -o -v >  tmp/aligned.txt')
word_align = pd.DataFrame(columns=('Gender', 'CnFile', 'EnFile', 'Alignment'))
with open('tmp/aligned.txt', 'r') as f:
    lines = f.readlines()
    for info, aligned in zip(pair_info, lines[:len(pair_info)]):
        info['Alignment'] = aligned.strip()
        word_align = word_align.append(info, ignore_index=True)
word_align.to_csv('csv/dataset_word_align.csv', index=False)
print_alignment(word_align)

import pandas as pd


cm = pd.read_csv('csv/transcription_cn_male.csv', header=0)
cf = pd.read_csv('csv/transcription_cn_female.csv', header=0)
em = pd.read_csv('csv/transcription_en_male.csv', header=0)
ef = pd.read_csv('csv/transcription_en_female.csv', header=0)
pairs = pd.read_csv('csv/pairs.csv', header=0)

f = pd.DataFrame(columns=['Gender', 'CnText', 'EnText', 'CnFile', 'EnFile'])


def add(gender, cnidx, enidx):
    global f
    if gender == 'Male':
        cnfile = '%d.wav'%cnidx
        cntext = cm[cm['Filename'].isin([cnfile])].iloc[0]['Text']
        enfile = '%d.wav'%enidx
        entext = em[em['Filename'].isin([enfile])].iloc[0]['Text']
        f = f.append({'Gender': 'male', 'CnText': cntext, 'EnText': entext,
                      'CnFile': cnfile, 'EnFile': enfile}, ignore_index=True)
    else:
        cnfile = '%d.wav'%cnidx
        cntext = cf[cf['Filename'].isin([cnfile])].iloc[0]['Text']
        enfile = '%d.wav'%enidx
        entext = ef[ef['Filename'].isin([enfile])].iloc[0]['Text']
        f = f.append({'Gender': 'female', 'CnText': cntext, 'EnText': entext,
                      'CnFile': cnfile, 'EnFile': enfile}, ignore_index=True)


for _, row in pairs.iterrows():
    add(row['Gender'], row['CnIdx'], row['EnIdx'])
f.to_csv('csv/dataset_meta.csv', index=False)

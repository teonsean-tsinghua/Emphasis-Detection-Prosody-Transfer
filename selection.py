import pandas as pd
import re


cm = pd.read_csv('csv/transcription_cn_male.csv', header=0)
cf = pd.read_csv('csv/transcription_cn_female.csv', header=0)
em = pd.read_csv('csv/transcription_en_male.csv', header=0)
ef = pd.read_csv('csv/transcription_en_female.csv', header=0)
pairs = pd.read_csv('csv/pairs.csv', header=0)

f = pd.DataFrame(columns=['Gender', 'CnText', 'EnText', 'CnFile', 'EnFile'])


def extend(text):
    pat_is = re.compile("(it|he|she|that|this|there|here|what|how|where|when|why|which)(\'s)", re.I)
    # to find the 's following the letters
    pat_s = re.compile("(?<=[a-zA-Z])\'s")
    # to find the ' following the words ending by s
    pat_s2 = re.compile("(?<=s)\'s?")
    # to find the abbreviation of not
    pat_not = re.compile("(?<=[a-zA-Z])n\'t")
    # to find the abbreviation of would
    pat_would = re.compile("(?<=[a-zA-Z])\'d")
    # to find the abbreviation of will
    pat_will = re.compile("(?<=[a-zA-Z])\'ll")
    # to find the abbreviation of am
    pat_am = re.compile("(?<=[I|i])\'m")
    # to find the abbreviation of are
    pat_are = re.compile("(?<=[a-zA-Z])\'re")
    # to find the abbreviation of have
    pat_ve = re.compile("(?<=[a-zA-Z])\'ve")

    new_text = pat_is.sub(r"\1 is", text)
    new_text = pat_s.sub("", new_text)
    new_text = pat_s2.sub("", new_text)
    new_text = pat_not.sub("n not", new_text)
    new_text = pat_would.sub(" would", new_text)
    new_text = pat_will.sub(" will", new_text)
    new_text = pat_am.sub(" am", new_text)
    new_text = pat_are.sub(" are", new_text)
    new_text = pat_ve.sub(" have", new_text)
    new_text = new_text.replace('\'', ' ')
    return new_text.replace('"', '').replace('?', '')


def add(gender, cnidx, enidx):
    global f
    if gender == 'Male':
        cnfile = '%d.wav'%cnidx
        cntext = cm[cm['Filename'].isin([cnfile])].iloc[0]['Text']
        enfile = '%d.wav'%enidx
        entext = extend(em[em['Filename'].isin([enfile])].iloc[0]['Text'])
        f = f.append({'Gender': 'male', 'CnText': cntext, 'EnText': entext,
                      'CnFile': cnfile, 'EnFile': enfile}, ignore_index=True)
    else:
        cnfile = '%d.wav'%cnidx
        cntext = cf[cf['Filename'].isin([cnfile])].iloc[0]['Text']
        enfile = '%d.wav'%enidx
        entext = extend(ef[ef['Filename'].isin([enfile])].iloc[0]['Text'])
        f = f.append({'Gender': 'female', 'CnText': cntext, 'EnText': entext,
                      'CnFile': cnfile, 'EnFile': enfile}, ignore_index=True)


for _, row in pairs.iterrows():
    add(row['Gender'], row['CnIdx'], row['EnIdx'])
f.to_csv('csv/dataset_meta.csv', index=False)

import pandas as pd


cm = pd.read_csv('csv/recognized_cn_male.csv', header=0)
cf = pd.read_csv('csv/recognized_cn_female.csv', header=0)
em = pd.read_csv('csv/recognized_en_male.csv', header=0)
ef = pd.read_csv('csv/recognized_en_female.csv', header=0)

f = pd.DataFrame(columns=['Gender', 'CnText', 'EnText', 'CnFile', 'EnFile'])


def add(gender, cnidx, enidx):
    global f
    if gender == 'male':
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


add('male', 49, 70)
add('male', 59, 83)
add('male', 82, 110)
add('male', 126, 175)
add('female', 26, 33)
add('female', 27, 34)
add('female', 29, 38)
add('female', 37, 50)
f.to_csv('csv/dataset_meta.csv', index=False)

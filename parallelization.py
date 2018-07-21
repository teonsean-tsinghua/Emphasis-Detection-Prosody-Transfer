import os
import pandas as pd
import subprocess
import xml.etree.ElementTree as et


try:
    os.mkdir('tmp')
except FileExistsError:
    pass

path = os.popen('pwd').readlines()[0].strip()
meta = pd.read_csv('text/dataset_meta.csv', header=0)
align = pd.DataFrame(columns=('Language', 'Gender', 'Filename', 'Active time', 'Alignment'))


def translate(lang, gend, name):
    global align
    tree = et.parse('./tmp/palign.xra')
    doc = tree.getroot()
    active_time = []
    alignment = []
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
    align = align.append({'Language': lang, 'Gender': gend, 'Filename': name,
                  'Active time': active_time, 'Alignment': alignment}, ignore_index=True)


for i, row in meta.iterrows():
    print('\n\n\n=================================================\nprocessing %d/%d examples.' % (i+1, len(meta)))
    commands = []
    commands.append('cd %s' % path)
    commands.append('echo %s | ' % row['CnText'] +
                    'python ./SPPAS/sppas/bin/normalize.py ' +
                    '-r ./SPPAS/resources/vocab/cmn.vocab ' +
                    '> ./tmp/text.txt')
    commands.append('python ./SPPAS/sppas/bin/wavsplit.py ' +
                    '-w ./audio/segment/cn/%s/%s ' % (row['Gender'], row['CnFile']) +
                    '-t ./tmp/text.txt ' +
                    '-p ./tmp/segm.xra')
    commands.append('python ./SPPAS/sppas/bin/normalize.py ' +
                    '-i ./tmp/segm.xra ' +
                    '-r ./SPPAS/resources/vocab/cmn.vocab ' +
                    '-o ./tmp/norm.xra')
    commands.append('python ./SPPAS/sppas/bin/phonetize.py ' +
                    '-i ./tmp/norm.xra ' +
                    '-r ./SPPAS/resources/dict/cmn.dict ' +
                    '-o ./tmp/phon.xra')
    commands.append('python ./SPPAS/sppas/bin/alignment.py ' +
                    '-w ./audio/segment/cn/%s/%s ' % (row['Gender'], row['CnFile']) +
                    '-i ./tmp/phon.xra ' +
                    '-I ./tmp/norm.xra ' +
                    '-r ./SPPAS/resources/models/models-cmn ' +
                    '-o ./tmp/palign.xra')
    subprocess.call(' && '.join(commands), shell=True)
    translate('cn', row['Gender'], row['CnFile'])
    commands = []
    commands.append('cd %s' % path)
    commands.append('echo %s | ' % row['EnText'] +
                    'python ./SPPAS/sppas/bin/normalize.py ' +
                    '-r ./SPPAS/resources/vocab/eng.vocab ' +
                    '> ./tmp/text.txt')
    commands.append('python ./SPPAS/sppas/bin/wavsplit.py ' +
                    '-w ./audio/segment/en/%s/%s ' % (row['Gender'], row['EnFile']) +
                    '-t ./tmp/text.txt ' +
                    '-p ./tmp/segm.xra')
    commands.append('python ./SPPAS/sppas/bin/normalize.py ' +
                    '-i ./tmp/segm.xra ' +
                    '-r ./SPPAS/resources/vocab/eng.vocab ' +
                    '-o ./tmp/norm.xra')
    commands.append('python ./SPPAS/sppas/bin/phonetize.py ' +
                    '-i ./tmp/norm.xra ' +
                    '-r ./SPPAS/resources/dict/eng.dict ' +
                    '-o ./tmp/phon.xra')
    commands.append('python ./SPPAS/sppas/bin/alignment.py ' +
                    '-w ./audio/segment/en/%s/%s ' % (row['Gender'], row['EnFile']) +
                    '-i ./tmp/phon.xra ' +
                    '-I ./tmp/norm.xra ' +
                    '-r ./SPPAS/resources/models/models-eng ' +
                    '-o ./tmp/palign.xra')
    subprocess.call(' && '.join(commands), shell=True)
    translate('en', row['Gender'], row['EnFile'])
align.to_csv('./text/dataset_time_align.csv', index=False)
try:
    os.rmdir('./tmp/')
except BaseException:
    pass

import pandas as pd
import xml.etree.ElementTree as et


meta = pd.read_csv('text/dataset_meta.csv', header=0)
align = pd.DataFrame(columns=('Language', 'Gender', 'Filename', 'Active time', 'Alignment'))


def translate(lang, gend, name):
    global align
    tree = et.parse('./tmp/%s_%s_%s.palign.xra' % (lang, gend, name))
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
    translate('cn', row['Gender'], row['CnFile'])
    translate('en', row['Gender'], row['EnFile'])
align.to_csv('text/dataset_time_align.csv', index=False)

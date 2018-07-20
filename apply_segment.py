from pydub import AudioSegment
import pandas as pd
import os

def apply_segment(audioname):
    try:
        os.mkdir('audio/segment/%s' % audioname)
        os.mkdir('audio/segment/%s/male' % audioname)
        os.mkdir('audio/segment/%s/female' % audioname)
    except FileExistsError:
        pass
    segments = pd.DataFrame(pd.read_csv('text/%s.csv' % audioname, header=None, sep='\t'))
    segments[1] = (segments[1] * 1000).astype(dtype=int, copy=False)
    segments[2] = (segments[2] * 1000).astype(dtype=int, copy=False)
    audio = AudioSegment.from_wav('audio/original/%s.wav' % audioname)
    male_cnt = 0
    female_cnt = 0
    for i, row in segments.iterrows():
        if row[0] == 'Male':
            (audio[row[1]:row[2]]).export('audio/segment/%s/male/%d.wav' % (audioname, male_cnt), format='wav')
            male_cnt += 1
        elif row[0] == 'Female':
            (audio[row[1]:row[2]]).export('audio/segment/%s/female/%d.wav' % (audioname, female_cnt), format='wav')
            female_cnt += 1


apply_segment('cn')
apply_segment('en')
from aip import AipSpeech
import pandas as pd
import os

APP_ID = '11543212'
API_KEY = 'jmmOKK8Gkzv6UMTa3nNR7wfv'
SECRET_KEY = 'HCkGzPGEY1d32HZ50eApvnTkd6X53qqN'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
text = pd.DataFrame(columns=('Language', 'Gender', 'Filename', 'Text'))


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


def speech2text(language, gender):
    global text
    dir_path = 'audio/segment/%s/%s/' % (language, gender)
    filenames = os.listdir(dir_path)
    for i, filename in enumerate(filenames):
        print('%d/%d' % (i, len(filenames)))
        re = client.asr(get_file_content(dir_path + filename), 'wav', 16000, {'dev_pid': 1536 if language == 'cn' else 1737})
        if re['err_no'] == 0:
            text = text.append({'Language': language, 'Gender': gender,
                                'Filename': filename, 'Text': re['result']}, ignore_index=True)
        else:
            print(re)

try:
    speech2text('cn', 'male')
    speech2text('cn', 'female')
    speech2text('en', 'male')
    speech2text('en', 'female')
except BaseException as e:
    print(e)
finally:
    text.to_csv('text/recognized_text.csv')
from aip import AipSpeech
import pandas as pd
import os
import time
import sys
import threading

""" 你的 APPID AK SK """
APP_ID = '11543212'
API_KEY = 'jmmOKK8Gkzv6UMTa3nNR7wfv'
SECRET_KEY = 'HCkGzPGEY1d32HZ50eApvnTkd6X53qqN'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
text = pd.DataFrame(columns=('Language', 'Gender', 'Filename', 'Text'))
threads = []
lock = threading.Lock()
lock2 = threading.Lock()
cnt = 0


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


def speech2text(language, gender):
    def recognize(path):
        global text, cnt
        re = client.asr(get_file_content(path), 'wav', 16000, {'dev_pid': 1536 if language == 'cn' else 1737})
        if re['err_no'] == 0:
            lock.acquire()
            text = text.append({'Language': language, 'Gender': gender,
                                'Filename': filename, 'Text': re['result']}, ignore_index=True)
            lock.release()
            lock2.acquire()
            print('%d finished' % cnt)
            cnt += 1
            lock2.release()

    dir_path = 'audio/segment/%s/%s/' % (language, gender)
    filenames = os.listdir(dir_path)
    for i, filename in enumerate(filenames):
        time.sleep(0.01)
        t = threading.Thread(target=recognize, args=(dir_path + filename,))
        t.start()
        threads.append(t)

try:
    speech2text('cn', 'male')
    speech2text('cn', 'female')
    speech2text('en', 'male')
    speech2text('en', 'female')
except BaseException as e:
    print(e)
    for t in threads:
        t.join()
    text.to_csv('text/recognized_text.csv')
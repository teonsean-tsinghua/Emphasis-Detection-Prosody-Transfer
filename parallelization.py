import os
import pandas as pd

try:
    os.mkdir('tmp')
except FileExistsError:
    pass

dataset = pd.read_csv('text/dataset_meta.csv', header=0)
for _, row in dataset.iterrows():
    os.system('echo "%s"' % row['CnText'])

import os
import pandas as pd

f = pd.DataFrame(columns=['Gender', 'CnText', 'EnText', 'CnPath', 'EnPath'])
f.to_csv('text/dataset_meta.csv', index=False)
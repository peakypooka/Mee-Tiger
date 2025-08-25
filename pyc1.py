import pandas as pd
from pycaret.classification import *

file_path = '/Users/peakypooka/Library/VSCode_Backup/Dev_Team/Mee-Tiger/src/train/data/d1.csv'

def load_data(file_path):
    return pd.read_csv(file_path)






s = setup(
    data = load_data(file_path),
    target = 'target',
    session_id = 123,
    normalize = True,
    train_size = 0.8,
)
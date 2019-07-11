import matplotlib.pyplot as plt
import seaborn; seaborn.set()
import os
import pandas as pd
import PyQt5
%matplotlib qt

# os.chdir("C:/Lin/Data science/Py training/Python DS handbook")

DATA_FOLDER = 'C:/Users/lind/Coursera/Advanced machine learning - Kaggle/Final project'

transactions    = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv.gz'))
items           = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))
item_categories = pd.read_csv(os.path.join(DATA_FOLDER, 'item_categories.csv'))
shops           = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))

transactions['year'] = pd.DatetimeIndex(transactions['date']).year

import matplotlib.pyplot as plt
import seaborn; seaborn.set()
import os
import pandas as pd
import numpy as np
import PyQt5
import statsmodels.api as sm
from pylab import rcParams
%matplotlib qt

# os.chdir("C:/Lin/Data science/Py training/Python DS handbook")

# DATA_FOLDER = 'C:/Users/lind/Coursera/Advanced machine learning - Kaggle/Final project'

DATA_FOLDER = 'C:/Lin/Data science/Github repo/Coursera/Advanced machine learning - Kaggle/Final project'

transactions    = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv.gz'))
items           = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))
item_categories = pd.read_csv(os.path.join(DATA_FOLDER, 'item_categories.csv'))
shops           = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))

# Creating year, month, day variable from date
transactions['date'] = pd.to_datetime(transactions['date'], format='%d.%m.%Y')

transactions['year'] = pd.DatetimeIndex(transactions['date']).year
transactions['month'] = pd.DatetimeIndex(transactions['date']).month
transactions['day'] = pd.DatetimeIndex(transactions['date']).day

# Creating a sales column for each day
transactions['sales'] = transactions['item_price'] * transactions['item_cnt_day']
transactions = pd.merge(transactions, items, on='item_id')
# The time series range
print('Timeseries start from ' + str(transactions['date'].min()) + ', finish on ' + str(transactions['date'].max()))
# Drop item name column, consider it as un-useful for now
transactions = transactions.drop(columns = 'item_name')

# Sort by date
transactions = transactions.sort_values('date')

# check missing value
transactions.isnull().sum()

# Look at the overall sales first
overall = transactions.groupby('date')['sales'].sum().reset_index()

overall = overall.set_index('date')
overall.index

# Look at the overall sales time series
overall.plot(figsize=(15, 6))
plt.show()

# Time-series docomposition to decompose into trend, seasonality and noise
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(overall, model='additive')
fig = decomposition.plot()
plt.show()













# end of script

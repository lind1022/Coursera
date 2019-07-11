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

# Creating year, month, day variable from date
transactions['year'] = pd.DatetimeIndex(transactions['date']).year
transactions['month'] = pd.DatetimeIndex(transactions['date']).month
transactions['day'] = pd.DatetimeIndex(transactions['date']).day

# Get sales value, price * count
transactions['sales'] = transactions['item_price'] * transactions['item_cnt_day']


# Question 1: what was the maximum total revenue among all the shops in September, 2014?
# Subset the data to September 2014
trans_interest = transactions[(transactions.year == 2014) & (transactions.month == 9)]
# Get total sales value for the period of interest
total_sales = trans_interest['sales'].sum() # Q1 ans: 116282853


# Question 2: what item category generated the highest revenue in summer 2014?
# Merge category id to transaction data
transactions = pd.merge(transactions, items, on='item_id')
# Subset transactions to summer period
summer_trans = transactions[(transactions.year == 2014) & (transactions.month >= 6) & (transactions.month <= 8)]

item_sales = summer_trans.groupby('item_category_id')['sales'].sum()
item_sales.sort_values(ascending = False)


# Question 3: How many items are there, such that their price stays constant (to the best of our knowledge) during the whole period of time?




















# haha

import matplotlib.pyplot as plt
import seaborn; seaborn.set()
import os
import pandas as pd
import numpy as np
import PyQt5
%matplotlib qt

# os.chdir("C:/Lin/Data science/Py training/Python DS handbook")

DATA_FOLDER = 'C:/Users/lind/Coursera/Advanced machine learning - Kaggle/Final project'

trans = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv.gz'))
items = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))
item_categories = pd.read_csv(os.path.join(DATA_FOLDER, 'item_categories.csv'))
shops = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))

# Creating year, month, day variable from date
trans['date'] = pd.to_datetime(trans['date'], format='%d.%m.%Y')

trans['year'] = pd.DatetimeIndex(trans['date']).year
trans['month'] = pd.DatetimeIndex(trans['date']).month
trans['day'] = pd.DatetimeIndex(trans['date']).day

# Get sales value, price * count
trans['sales'] = trans['item_price'] * trans['item_cnt_day']


# Question 1: what was the maximum total revenue among all the shops in September, 2014?
# Subset the data to September 2014
trans = pd.merge(trans, shops, on='shop_id')
trans_interest = trans[(trans.year == 2014) & (trans.month == 9)]
# Get total sales value for the period of interest
max2014 = trans_interest.groupby('shop_name')['sales'].sum().sort_values(ascending = False)
max(max2014.values) #Q1 ans



# Question 2: what item category generated the highest revenue in summer 2014?
# Merge category id to transaction data
trans = pd.merge(trans, items, on='item_id')
# Subset transactions to summer period
summer_trans = trans[(trans.year == 2014) & (trans.month >= 6) & (trans.month <= 8)]

item_sales = summer_trans.groupby('item_category_id')['sales'].sum()
item_sales.sort_values(ascending = False) # Q2 ans


# Question 3: How many items are there, such that their price stays constant (to the best of our knowledge) during the whole period of time?
item_price_points = trans.groupby('item_id')['item_price'].nunique()
len(item_price_points[item_price_points == 1]) # Q3 ans 5926

# Question 4: what was the variance of the number of sold items per day sequence for the shop with shop_id = 25
# in December, 2014? Do not count the items, that were sold but returned back later.
shop_id = 25
# Subset to December 2014
dec_2014_trans = trans[(trans.year == 2014) & (trans.month == 12) & (trans.shop_id == shop_id)]

total_num_items_sold = np.array(dec_2014_trans.groupby('date')['item_cnt_day'].sum()) # YOUR CODE GOES HERE
days = np.array(dec_2014_trans['date'].drop_duplicates().sort_values())# YOUR CODE GOES HERE

# Plot it
plt.plot(days, total_num_items_sold)
plt.ylabel('Num items')
plt.xlabel('Day')
plt.title("Daily revenue for shop_id = 25")
plt.show()

# Q4 ans Unbiased variance
np.var(total_num_items_sold, ddof = 1)


%matplotlib qt

import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
uniform_data = np.random.rand(10, 12)
ax = sns.heatmap(uniform_data)



def render(target_var):
    return f'this is the {self.target_var}'



# haha

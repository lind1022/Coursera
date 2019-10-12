import matplotlib.pyplot as plt
import seaborn; seaborn.set()
import os
import pandas as pd
import numpy as np
import PyQt5
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import scipy.stats as scs

from pylab import rcParams
import itertools
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
%matplotlib qt


"""
Some Tips

A good exercise is to reproduce previous_value_benchmark. As the name suggest -
in this benchmark for the each shop/item pair our predictions are just monthly
sales from the previous month, i.e. October 2015.

The most important step at reproducing this score is correctly aggregating daily
data and constructing monthly sales data frame.
You need to get lagged values, fill NaNs with zeros and clip the values into
[0,20] range. If you do it correctly, you'll get precisely 1.16777 on the public
leaderboard.

Generating features like this is a necessary basis for more complex models.
Also, if you decide to fit some model, don't forget to clip the target into
[0,20] range, it makes a big difference.

You can get a rather good score after creating some lag-based features like in
 advice from previous week and feeding them into gradient boosted trees model.

Apart from item/shop pair lags you can try adding lagged values of total shop or
 total item sales (which are essentially mean-encodings). All of that is going
 to add some new information.
"""


# DATA_FOLDER = 'C:/Users/lind/Coursera/Advanced machine learning - Kaggle/Final project'

DATA_FOLDER = 'C:/Lin/Data science/Github repo/Coursera/Advanced machine learning - Kaggle/Final project'

trans           = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv.gz'))
items           = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))
item_categories = pd.read_csv(os.path.join(DATA_FOLDER, 'item_categories.csv'))
shops           = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))

test           = pd.read_csv(os.path.join(DATA_FOLDER, 'test.csv.gz'))
sample         = pd.read_csv(os.path.join(DATA_FOLDER, 'sample_submission.csv.gz'))

os.chdir(DATA_FOLDER)

# Creating year, month, day variable from date
trans['date'] = pd.to_datetime(trans['date'], format='%d.%m.%Y')

trans['year'] = pd.DatetimeIndex(trans['date']).year
trans['month'] = pd.DatetimeIndex(trans['date']).month
trans['day'] = pd.DatetimeIndex(trans['date']).day

# Creating a sales column for each day
trans['sales'] = trans['item_price'] * trans['item_cnt_day']
trans = pd.merge(trans, items, on='item_id')
# The time series range
print('Timeseries start from ' + str(trans['date'].min()) + ', finish on ' + str(trans['date'].max()))
# Drop item name column, consider it as un-useful for now
trans = trans.drop(columns = 'item_name')

# Sort by date
trans = trans.sort_values('date')

# check missing value and unique values for each column
trans.isnull().sum()
trans.nunique()

# Control for data leakage, only use item and store that appear in the test dataset.
test_shop_ids = test['shop_id'].unique()
test_item_ids = test['item_id'].unique()

trans = trans[trans['shop_id'].isin(test_shop_ids)]
trans = trans[trans['item_id'].isin(test_item_ids)]

# Aggregate to montly data
ts = trans.groupby(['year', 'month', 'shop_id', 'item_id', 'item_category_id'])[['item_cnt_day', 'sales']].sum().reset_index()


# To mimic the real behavior of the data we have to create the missing records
# from the loaded dataset, so for each month we need to create the missing records
# for each shop and item, since we don't have data for them I'll replace them with 0.
shop_ids = trans['shop_id'].unique()
item_ids = trans['item_id'].unique()
days = len(trans['date_block_num'].unique())
empty_df = []
for i in range(days):
    for shop in shop_ids:
        for item in item_ids:
            empty_df.append([i, shop, item])

empty_df = pd.DataFrame(empty_df, columns=['date_block_num','shop_id','item_id'])



###########################################################
# Creating the prediction baseline with Oct 15 sales
###########################################################

oct_15 = ts[(ts.year == 2015) & (ts.month == 10)][['shop_id', 'item_id', 'item_cnt_day']]

pred_test = pd.merge(test, oct_15, on=['shop_id', 'item_id'], how='left')
pred_test['item_cnt_day'] = pred_test['item_cnt_day'].fillna(0)

# Clip sales values into the [0, 20] range
pred_test['item_cnt_day'][pred_test['item_cnt_day'] < 0] = 0
pred_test['item_cnt_day'][pred_test['item_cnt_day'] > 20] = 20

# File to submit
submit = pred_test[['ID', 'item_cnt_day']].rename(columns = {'item_cnt_day': 'item_cnt_month'})
submit.to_csv("submission.csv", index = False)


# end of script

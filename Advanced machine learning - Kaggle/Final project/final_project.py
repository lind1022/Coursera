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
from itertools import product

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

If you successfully made use of previous advises, it's time to move forward and
incorporate some new knowledge from week 4. Here are several things you can do:

Try to carefully tune hyper parameters of your models, maybe there is a better
set of parameters for your model out there. But don't spend too much time on it.
Try ensembling. Start with simple averaging of linear model and gradient boosted
trees like in programming assignment notebook. And then try to use stacking.
Explore new features! There is a lot of useful information in the data: text
descriptions, item categories, seasonal trends.
"""

# Create a function to downcast data to save storage
def downcast_dtypes(df):
    '''
        Changes column types in the dataframe:

                `float64` type to `float32`
                `int64`   type to `int32`
    '''

    # Select columns to downcast
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols =   [c for c in df if df[c].dtype == "int64"]

    # Downcast
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols]   = df[int_cols].astype(np.int32)

    return df

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
# trans['date'] = pd.to_datetime(trans['date'], format='%d.%m.%Y')
#
# trans['year'] = pd.DatetimeIndex(trans['date']).year
# trans['month'] = pd.DatetimeIndex(trans['date']).month
# trans['day'] = pd.DatetimeIndex(trans['date']).day

# Creating a sales column for each day
trans = pd.merge(trans, items, on='item_id')
# The time series range
# print('Timeseries start from ' + str(trans['date'].min()) + ', finish on ' + str(trans['date'].max()))

# Sort by date
trans = trans.sort_values('date_block_num')

# Drop item name column, consider it as un-useful for now
trans = trans.drop(columns = ['item_name', 'date'])

# Control for data leakage, only use item and store that appear in the test dataset.
test_shop_ids = test['shop_id'].unique()
test_item_ids = test['item_id'].unique()

# Keep data that has shop and item in the test dataset
trans = trans[trans['shop_id'].isin(test_shop_ids)]
trans = trans[trans['item_id'].isin(test_item_ids)]


index_cols = ['shop_id', 'item_id', 'date_block_num']

# For every month we create a grid from all shops/items combinations from that month
grid = []
for block_num in trans['date_block_num'].unique():
    cur_shops = trans.loc[trans['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = trans.loc[trans['date_block_num'] == block_num, 'item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

# Turn the grid into a dataframe
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)

# Aggregate to montly data by month, shop, item
gb = trans.groupby(['date_block_num', 'shop_id', 'item_id']).agg(shop_item_month=('item_cnt_day', 'sum')).reset_index()
train = pd.merge(grid, gb, how='left', on=['date_block_num', 'shop_id', 'item_id']).fillna(0)

# Aggregate to item-month
gb = trans.groupby(['date_block_num', 'item_id']).agg(item_month=('item_cnt_day', 'sum')).reset_index()
train = pd.merge(train, gb, how='left', on=['date_block_num', 'item_id']).fillna(0)

# Aggregate to shop-month
gb = trans.groupby(['date_block_num', 'shop_id']).agg(shop_month=('item_cnt_day', 'sum')).reset_index()
train = pd.merge(train, gb, how='left', on=['date_block_num', 'shop_id']).fillna(0)

train = downcast_dtypes(train)
del gb

# trans = downcast_dtypes(trans)

#############################
# Adding lag based features
#############################

# Creating a lag term
cols_to_rename = ['shop_item_month', 'item_month', 'shop_month']

shift_range = [1, 2, 3, 4, 5, 12]


for month_shift in shift_range:
    train_shift = train[index_cols + cols_to_rename].copy()

    train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift

    foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x
    train_shift = train_shift.rename(columns=foo)
    train = pd.merge(train, train_shift, on=index_cols, how='left').fillna(0)

del train_shift

# Don't use old data from year 2013
trans = trans[trans['date_block_num'] >= 12]

# List of all lagged features
fit_cols = [col for col in trans.columns if col[-1] in [str(item) for item in shift_range]]
# We will drop these at fitting stage
to_drop_cols = list(set(list(trans.columns)) - (set(fit_cols)|set(index_cols))) + ['date_block_num']

# Category for each item
item_category_mapping = items[['item_id','item_category_id']].drop_duplicates()

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

# Merge the train set with the complete set (missing records will be filled with 0).
trans = pd.merge(empty_df, trans, on=index_cols, how='left').fillna(0)


trans = pd.merge(trans, item_category_mapping, how='left', on='item_id')
trans = downcast_dtypes(trans)
gc.collect();











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

kaggle competitions submit -c competitive-data-science-predict-future-sales -f submission.csv -m "Message"

# end of script

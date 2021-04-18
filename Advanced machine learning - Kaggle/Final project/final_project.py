import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import os
import pandas as pd
import numpy as np
import PyQt5
import statsmodels.api as sm
import scipy.stats as scs
from itertools import product
import gc

import itertools
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time

from catboost import *
import catboost
from catboost import Pool
from catboost import CatBoostClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance

packages = [np, catboost, pd, sns, sklearn, xgboost, statsmodels]
for package in packages:
    print(f'{package.__name__} {package.__version__}')


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
# DATA_FOLDER = 'C:/Lin/Data science/Github repo/Coursera/Advanced machine learning - Kaggle/Final project'

# Mac directory
DATA_FOLDER = '/Users/linding/github/Coursera/Advanced machine learning - Kaggle/Final project'
os.chdir(DATA_FOLDER)

trans           = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv.gz'))
items           = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))
item_categories = pd.read_csv(os.path.join(DATA_FOLDER, 'item_categories.csv'))
shops           = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))

test           = pd.read_csv(os.path.join(DATA_FOLDER, 'test.csv.gz')).set_index('ID')
# sample         = pd.read_csv(os.path.join(DATA_FOLDER, 'sample_submission.csv.gz'))


# Creating year, month, day variable from date
# trans['date'] = pd.to_datetime(trans['date'], format='%d.%m.%Y')
#
# trans['year'] = pd.DatetimeIndex(trans['date']).year
# trans['month'] = pd.DatetimeIndex(trans['date']).month
# trans['day'] = pd.DatetimeIndex(trans['date']).day


# The time series range
print('Timeseries start from ' + str(trans['date'].min()) + ', finish on ' + str(trans['date'].max()))

#######
# EDA #
#######
# Grouping data for EDA.
# Category for each item
item_category_mapping = items[['item_id','item_category_id']].drop_duplicates()
trans = pd.merge(trans, item_category_mapping, how='left', on='item_id')


# Sales by item category
# gp_category_mean = trans.groupby(['item_category_id'], as_index=False)['item_cnt_day'].mean()
# gp_category_sum = trans.groupby(['item_category_id'], as_index=False)['item_cnt_day'].sum()

# Sales by shop
# gp_shop_mean = trans.groupby(['shop_id'], as_index=False)['item_cnt_day'].mean()
# gp_shop_sum = trans.groupby(['shop_id'], as_index=False)['item_cnt_day'].sum()


# f, axes = plt.subplots(2, 1, figsize=(22, 10), sharex=True)
# sns.barplot(x="item_category_id", y="item_cnt_day", data=gp_category_mean, ax=axes[0], palette="rocket").set_title("Monthly mean")
# sns.barplot(x="item_category_id", y="item_cnt_day", data=gp_category_sum, ax=axes[1], palette="rocket").set_title("Monthly sum")
# plt.show()
#
# f, axes = plt.subplots(2, 1, figsize=(22, 10), sharex=True)
# sns.barplot(x="shop_id", y="item_cnt_day", data=gp_shop_mean, ax=axes[0], palette="rocket").set_title("Monthly mean")
# sns.barplot(x="shop_id", y="item_cnt_day", data=gp_shop_sum, ax=axes[1], palette="rocket").set_title("Monthly sum")
# plt.show()

########################
# Feature Engineering
########################

# Checking for outliers
plt.figure(figsize=(10,4))
plt.xlim(-100, 3000)
# sns.boxplot(x=trans.item_cnt_day)

plt.figure(figsize=(10,4))
plt.xlim(trans.item_price.min(), trans.item_price.max()*1.1)
# sns.boxplot(x=trans.item_price)

# Removing outliers
trans = trans[trans.item_price<100000]
trans = trans[trans.item_cnt_day<1001]

# One item with negative price, fill with median
median = trans[(trans.shop_id==32)&(trans.item_id==2973)&(trans.date_block_num==4)&(trans.item_price>0)].item_price.median()
trans.loc[trans.item_price<0, 'item_price'] = median

# Several duplicated shops, fix both train and test
trans.loc[trans.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57

trans.loc[trans.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58

trans.loc[trans.shop_id == 10, 'shop_id'] = 11
test.loc[test.shop_id == 10, 'shop_id'] = 11


# Item/Shop/Category features
shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
shops['city_code'] = LabelEncoder().fit_transform(shops['city'])
shops = shops[['shop_id','city_code']]

item_categories['split'] = item_categories['item_category_name'].str.split('-')
item_categories['type'] = item_categories['split'].map(lambda x: x[0].strip())
item_categories['type_code'] = LabelEncoder().fit_transform(item_categories['type'])
# if subtype is nan then type
item_categories['subtype'] = item_categories['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
item_categories['subtype_code'] = LabelEncoder().fit_transform(item_categories['subtype'])
item_categories = item_categories[['item_category_id','type_code', 'subtype_code']]

matrix = []
cols = ['date_block_num','shop_id','item_id']
for i in range(34):
    sales = trans[trans.date_block_num==i]
    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))

matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)
matrix.sort_values(cols,inplace=True)

trans['revenue'] = trans['item_price'] *  trans['item_cnt_day']

group = trans.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})
group.columns = ['item_cnt_month']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=cols, how='left')
matrix['item_cnt_month'] = (matrix['item_cnt_month']
                                .fillna(0)
                                .clip(0,20) # NB clip target here
                                .astype(np.float16))

#
# # Sort by date
# trans = trans.sort_values('date_block_num')
#
# # Drop item name column, consider it as un-useful for now
# trans = trans.drop(columns = ['date'])
#
# # Control for data leakage, only use item and store that appear in the test dataset.
# test_shop_ids = test['shop_id'].unique()
# test_item_ids = test['item_id'].unique()
#
# # Keep data that has shop and item in the test dataset
# trans = trans[trans['shop_id'].isin(test_shop_ids)]
# trans = trans[trans['item_id'].isin(test_item_ids)]
#
# index_cols = ['shop_id', 'item_id', 'date_block_num']
#
# # For every month we create a grid from all shops/items combinations from that month
# block_nums = trans['date_block_num'].unique()
# all_shops = trans.loc[:,'shop_id'].unique()
# all_items = trans.loc[:,'item_id'].unique()
# grid = np.array(list(product(*[test_shop_ids, test_item_ids, block_nums])),dtype='int32')
#
#
# # Turn the grid into a dataframe
# grid = pd.DataFrame(np.vstack(grid), columns = index_cols, dtype=np.int32)
# grid.shape
#
#
#
# # Aggregate to monthly data by month, shop, item
# col_order = ['date_block_num', 'shop_id', 'item_id', 'item_cnt_day', 'item_price']
# gb = trans.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False).agg({'item_cnt_day': 'sum', 'item_price': 'mean'})[col_order]
# gb.columns = ['date_block_num', 'shop_id', 'item_id', 'item_cnt_month', 'item_price']
#
# # plt.subplots(figsize=(22, 8))
# # sns.boxplot(gb['item_cnt_month'])
# # plt.show()
#
# gb['revenue'] = gb['item_price'] * gb['item_cnt_month']
#
# # Clip sales values into the [0, 20] range
# gb['item_cnt_month'][gb['item_cnt_month'] < 0] = 0
# gb['item_cnt_month'][gb['item_cnt_month'] > 20] = 20
#
# train = pd.merge(grid, gb, how='left', on=['date_block_num', 'shop_id', 'item_id']).fillna(0)
# del gb

# Create target variable
# train['target'] = train.sort_values('date_block_num').groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(-1)

# Test data set
test['date_block_num'] = 34
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)

matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
matrix.fillna(0, inplace=True) # 34 month


# need to add category, year and month mapping
matrix = pd.merge(matrix, item_category_mapping, on='item_id', how='left')
matrix = pd.merge(matrix, shops, on='shop_id', how='left')
matrix = pd.merge(matrix, item_categories, on='item_category_id', how='left')

# Number of days in the month
matrix['month'] = matrix['date_block_num'] % 12

days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31], index=range(0, 12))
matrix['days'] = matrix['month'].map(days).astype(np.int8)

matrix = downcast_dtypes(matrix)


#############################
# Adding lag based features
#############################

def lag_feature(df, lags, col):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df

matrix = lag_feature(matrix, [1,2,3,6,12], 'item_cnt_month')

# Monthly average
group = matrix.groupby(['date_block_num']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num'], how='left')
matrix['date_avg_item_cnt'] = matrix['date_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_avg_item_cnt')
matrix.drop(['date_avg_item_cnt'], axis=1, inplace=True)

# Monthly average per item
group = matrix.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_item_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')
matrix['date_item_avg_item_cnt'] = matrix['date_item_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1,2,3,6,12], 'date_item_avg_item_cnt')
matrix.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)

# Monthly average per shop
group = matrix.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_shop_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','shop_id'], how='left')
matrix['date_shop_avg_item_cnt'] = matrix['date_shop_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1,2,3,6,12], 'date_shop_avg_item_cnt')
matrix.drop(['date_shop_avg_item_cnt'], axis=1, inplace=True)

# Monthly average per category
group = matrix.groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_cat_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','item_category_id'], how='left')
matrix['date_cat_avg_item_cnt'] = matrix['date_cat_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_cat_avg_item_cnt')
matrix.drop(['date_cat_avg_item_cnt'], axis=1, inplace=True)

# Monthly average per shop category
group = matrix.groupby(['date_block_num', 'shop_id', 'item_category_id']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_shop_cat_avg_item_cnt']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
matrix['date_shop_cat_avg_item_cnt'] = matrix['date_shop_cat_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_shop_cat_avg_item_cnt')
matrix.drop(['date_shop_cat_avg_item_cnt'], axis=1, inplace=True)


# Monthly average per type & shop
group = matrix.groupby(['date_block_num', 'shop_id', 'type_code']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_shop_type_avg_item_cnt']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'type_code'], how='left')
matrix['date_shop_type_avg_item_cnt'] = matrix['date_shop_type_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_shop_type_avg_item_cnt')
matrix.drop(['date_shop_type_avg_item_cnt'], axis=1, inplace=True)


# Monthly average per subtype & shop
group = matrix.groupby(['date_block_num', 'shop_id', 'subtype_code']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_shop_subtype_avg_item_cnt']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'subtype_code'], how='left')
matrix['date_shop_subtype_avg_item_cnt'] = matrix['date_shop_subtype_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_shop_subtype_avg_item_cnt')
matrix.drop(['date_shop_subtype_avg_item_cnt'], axis=1, inplace=True)


# Monthly average per city
group = matrix.groupby(['date_block_num', 'city_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_city_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'city_code'], how='left')
matrix['date_city_avg_item_cnt'] = matrix['date_city_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_city_avg_item_cnt')
matrix.drop(['date_city_avg_item_cnt'], axis=1, inplace=True)


# Monthly average per city & item
group = matrix.groupby(['date_block_num', 'item_id', 'city_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_item_city_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id', 'city_code'], how='left')
matrix['date_item_city_avg_item_cnt'] = matrix['date_item_city_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_item_city_avg_item_cnt')
matrix.drop(['date_item_city_avg_item_cnt'], axis=1, inplace=True)



# Monthly average per item type
group = matrix.groupby(['date_block_num', 'type_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_type_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'type_code'], how='left')
matrix['date_type_avg_item_cnt'] = matrix['date_type_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_type_avg_item_cnt')
matrix.drop(['date_type_avg_item_cnt'], axis=1, inplace=True)


# Monthly average per subtype item
group = matrix.groupby(['date_block_num', 'subtype_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_subtype_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'subtype_code'], how='left')
matrix['date_subtype_avg_item_cnt'] = matrix['date_subtype_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_subtype_avg_item_cnt')
matrix.drop(['date_subtype_avg_item_cnt'], axis=1, inplace=True)


###################
# Trend Features
###################

# Average price for each item
group = trans.groupby(['item_id']).agg({'item_price': ['mean']})
group.columns = ['item_avg_item_price']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['item_id'], how='left')
matrix['item_avg_item_price'] = matrix['item_avg_item_price'].astype(np.float16)

# Average price in each month for each item
group = trans.groupby(['date_block_num','item_id']).agg({'item_price': ['mean']})
group.columns = ['date_item_avg_item_price']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')
matrix['date_item_avg_item_price'] = matrix['date_item_avg_item_price'].astype(np.float16)

lags = [1,2,3,4,5,6]
matrix = lag_feature(matrix, lags, 'date_item_avg_item_price')

for i in lags:
    matrix['delta_price_lag_'+str(i)] = \
        (matrix['date_item_avg_item_price_lag_'+str(i)] - matrix['item_avg_item_price']) / matrix['item_avg_item_price']

def select_trend(row):
    for i in lags:
        if row['delta_price_lag_'+str(i)]:
            return row['delta_price_lag_'+str(i)]
    return 0

matrix['delta_price_lag'] = matrix.apply(select_trend, axis=1)
matrix['delta_price_lag'] = matrix['delta_price_lag'].astype(np.float16)
matrix['delta_price_lag'].fillna(0, inplace=True)

# https://stackoverflow.com/questions/31828240/first-non-null-value-per-row-from-a-list-of-pandas-columns/31828559
# matrix['price_trend'] = matrix[['delta_price_lag_1','delta_price_lag_2','delta_price_lag_3']].bfill(axis=1).iloc[:, 0]
# Invalid dtype for backfill_2d [float16]

features_to_drop = ['item_avg_item_price', 'date_item_avg_item_price']
for i in lags:
    features_to_drop += ['date_item_avg_item_price_lag_'+str(i)]
    features_to_drop += ['delta_price_lag_'+str(i)]

matrix.drop(features_to_drop, axis=1, inplace=True)

# Last month shop revenue trend
group = trans.groupby(['date_block_num','shop_id']).agg({'revenue': ['sum']})
group.columns = ['date_shop_revenue']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','shop_id'], how='left')
matrix['date_shop_revenue'] = matrix['date_shop_revenue'].astype(np.float32)

group = group.groupby(['shop_id']).agg({'date_shop_revenue': ['mean']})
group.columns = ['shop_avg_revenue']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['shop_id'], how='left')
matrix['shop_avg_revenue'] = matrix['shop_avg_revenue'].astype(np.float32)

matrix['delta_revenue'] = (matrix['date_shop_revenue'] - matrix['shop_avg_revenue']) / matrix['shop_avg_revenue']
matrix['delta_revenue'] = matrix['delta_revenue'].astype(np.float16)

matrix = lag_feature(matrix, [1], 'delta_revenue')

matrix.drop(['date_shop_revenue','shop_avg_revenue','delta_revenue'], axis=1, inplace=True)


###################
# Other features ##
###################


# Months since the last sale for each shop/item pair and for item only.

cache = {}
matrix['item_shop_last_sale'] = -1
matrix['item_shop_last_sale'] = matrix['item_shop_last_sale'].astype(np.int8)
for idx, row in matrix.iterrows():
    key = str(row.item_id)+' '+str(row.shop_id)
    if key not in cache:
        if row.item_cnt_month!=0:
            cache[key] = row.date_block_num
    else:
        last_date_block_num = cache[key]
        matrix.at[idx, 'item_shop_last_sale'] = row.date_block_num - last_date_block_num
        cache[key] = row.date_block_num

cache = {}
matrix['item_last_sale'] = -1
matrix['item_last_sale'] = matrix['item_last_sale'].astype(np.int8)
for idx, row in matrix.iterrows():
    key = row.item_id
    if key not in cache:
        if row.item_cnt_month!=0:
            cache[key] = row.date_block_num
    else:
        last_date_block_num = cache[key]
        if row.date_block_num>last_date_block_num:
            matrix.at[idx, 'item_last_sale'] = row.date_block_num - last_date_block_num
            cache[key] = row.date_block_num

# Months since the first sale for each shop/item pair and for item only.
matrix['item_shop_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
matrix['item_first_sale'] = matrix['date_block_num'] - matrix.groupby('item_id')['date_block_num'].transform('min')

######################
# Final Preparation
######################



matrix = matrix[matrix.date_block_num > 11]

# Fill missing value with 0.
def fill_na(df):
    for col in df.columns:
        if ('_lag_' in col) & (df[col].isnull().any()):
            if ('item_cnt' in col):
                df[col].fillna(0, inplace=True)
    return df

matrix = fill_na(matrix)

matrix.to_pickle('data.pkl')
del matrix
del cache
del group
del items
del shops
del train
# leave test for submission
gc.collect();

data = pd.read_pickle('data.pkl')
data = data[[
    'date_block_num',
    'shop_id',
    'item_id',
    'item_cnt_month',
    'city_code',
    'item_category_id',
    'type_code',
    'subtype_code',
    'item_cnt_month_lag_1',
    'item_cnt_month_lag_2',
    'item_cnt_month_lag_3',
    'item_cnt_month_lag_6',
    'item_cnt_month_lag_12',
    'date_avg_item_cnt_lag_1',
    'date_item_avg_item_cnt_lag_1',
    'date_item_avg_item_cnt_lag_2',
    'date_item_avg_item_cnt_lag_3',
    'date_item_avg_item_cnt_lag_6',
    'date_item_avg_item_cnt_lag_12',
    'date_shop_avg_item_cnt_lag_1',
    'date_shop_avg_item_cnt_lag_2',
    'date_shop_avg_item_cnt_lag_3',
    'date_shop_avg_item_cnt_lag_6',
    'date_shop_avg_item_cnt_lag_12',
    'date_cat_avg_item_cnt_lag_1',
    'date_shop_cat_avg_item_cnt_lag_1',
    # 'date_shop_type_avg_item_cnt_lag_1',
    # 'date_shop_subtype_avg_item_cnt_lag_1',
    'date_city_avg_item_cnt_lag_1',
    'date_item_city_avg_item_cnt_lag_1',
    # 'date_type_avg_item_cnt_lag_1',
    # 'date_subtype_avg_item_cnt_lag_1',
    # 'delta_revenue_lag_1', # just added
    'delta_price_lag',
    # 'year', # just added
    'month',
    'days',
    'item_shop_last_sale',
    'item_last_sale',
    'item_shop_first_sale',
    'item_first_sale',
]]



# Train test split
X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month'].astype('float32')
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month'].astype('float32')
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

int_features = ['shop_id', 'item_id', 'month', 'item_category_id', 'city_code', 'type_code', 'subtype_code']


# XGBoost
xgb_model = XGBRegressor(
    max_depth=8,
    n_estimators=800,
    min_child_weight=300,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.3,
    seed=42)

xgb_model.fit(
    X_train,
    Y_train,
    eval_metric="rmse",
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
    verbose=True,
    early_stopping_rounds = 10)


xgb_val_pred = xgb_model.predict(X_valid).clip(0, 20)
xgb_test_pred = xgb_model.predict(X_test).clip(0, 20)
xgb_train_pred = xgb_model.predict(X_train).clip(0, 20)



print('Train rmse by xgboost:', np.sqrt(mean_squared_error(Y_train, xgb_train_pred)))
print('Validation rmse by xgboost:', np.sqrt(mean_squared_error(Y_valid, xgb_val_pred)))

############
# Catboost
############

# Integer features (used by catboost model).

X_train[int_features] = X_train[int_features].astype('int32')
X_valid[int_features] = X_valid[int_features].astype('int32')
X_test[int_features] = X_test[int_features].astype('int32')

pool = Pool(data=X_train, label=Y_train)
print(pool.get_feature_names())

cat_model = CatBoostRegressor(
    iterations=500,
    max_ctr_complexity=7,
    random_seed=0,
    od_type='Iter',
    od_wait=25,
    verbose=50,
    depth=7,
    l2_leaf_reg=5
)

cat_model.fit(
    X_train, Y_train,
    cat_features=int_features,
    eval_set=(X_valid, Y_valid)
)



cat_model.best_score_['validation']['RMSE']

catboost_train_pred = cat_model.predict(X_train)
catboost_val_pred = cat_model.predict(X_valid)
catboost_test_pred = cat_model.predict(X_test)

print('Train rmse by catboost:', np.sqrt(mean_squared_error(Y_train, catboost_train_pred)))
print('Validation rmse by catboost:', np.sqrt(mean_squared_error(Y_valid, catboost_val_pred)))

##################
# Random Forest
##################
rf_model = RandomForestRegressor(n_estimators=50,
                                 max_depth=10,
                                 random_state=0,
                                 n_jobs=-1)
rf_model.fit(X_train, Y_train)


rf_train_pred = rf_model.predict(X_train)
rf_val_pred = rf_model.predict(X_valid)
rf_test_pred = rf_model.predict(X_test)

print('Train rmse by random forest:', np.sqrt(mean_squared_error(Y_train, rf_train_pred)))
print('Validation rmse by random forest:', np.sqrt(mean_squared_error(Y_valid, rf_val_pred)))

#########################
# K-nearest neighbour
#########################

knn_train = X_train.drop(int_features, axis=1)
knn_valid = X_valid.drop(int_features, axis=1)
knn_test = X_test.drop(int_features, axis=1)

knn_scaler = MinMaxScaler()
knn_scaler.fit(knn_train)

knn_train = knn_scaler.transform(knn_train)
knn_valid = knn_scaler.transform(knn_valid)
knn_test = knn_scaler.transform(knn_test)

knn_model = KNeighborsRegressor(n_neighbors=9, leaf_size=13, n_jobs=-1)
print('Training start...')
ts = time.time()
knn_model.fit(knn_train, Y_train)
time.time() - ts
print('Done')


#############################
# Support Vector Machine
#############################

# One-hot encoding category features
enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(X_train[int_features]).toarray())

X_train_enc = X_train.join(enc_df)

#####################
# Ridge Regression
#####################
model = Ridge()
# Find best alpha with cross-validation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['alpha'] = np.arange(0, 1, 0.01)
# define search
search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# perform the search
ridge_model = search.fit(X_train, Y_train)

ridge_train_pred = ridge_model.predict(X_train)
ridge_val_pred = ridge_model.predict(X_valid)
ridge_test_pred = ridge_model.predict(X_test)

print('Train rmse by Ridge Regression:', np.sqrt(mean_squared_error(Y_train, ridge_train_pred)))
print('Validation rmse by Ridge Regression:', np.sqrt(mean_squared_error(Y_valid, ridge_val_pred)))



# Ensemble
first_level = pd.DataFrame(xgb_val_pred, columns = ['xgboost'])
first_level['catboost'] = catboost_val_pred
first_level['random_forest'] = rf_val_pred
first_level['label'] = Y_valid.values

first_level_test = pd.DataFrame(xgb_test_pred, columns=['xgboost'])
first_level_test['catboost'] = catboost_test_pred
first_level_test['random_forest'] = rf_test_pred
first_level_test.head()

# 2nd level model as linear regression
meta_model = LinearRegression(n_jobs=-1)

first_level.drop('label', axis=1, inplace=True)
meta_model.fit(first_level, Y_valid)

ensemble_pred = meta_model.predict(first_level)
final_predictions = meta_model.predict(first_level_test)

print('Train rmse:', np.sqrt(mean_squared_error(ensemble_pred, Y_valid)))

submission = pd.DataFrame({
    "ID": test.index,
    "item_cnt_month": rf_test_pred.clip(0., 20.)
})
submission.to_csv('submission.csv', index=False)


# end of script

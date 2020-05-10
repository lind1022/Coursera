import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import os
import pandas as pd
import numpy as np
import PyQt5
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import scipy.stats as scs
from itertools import product
import gc

from pylab import rcParams
import itertools
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic

from catboost import *
import catboost
from catboost import Pool
from catboost import CatBoostClassifier
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
trans['date'] = pd.to_datetime(trans['date'], format='%d.%m.%Y')

trans['year'] = pd.DatetimeIndex(trans['date']).year
trans['month'] = pd.DatetimeIndex(trans['date']).month
trans['day'] = pd.DatetimeIndex(trans['date']).day

# The time series range
print('Timeseries start from ' + str(trans['date'].min()) + ', finish on ' + str(trans['date'].max()))

#######
# EDA #
#######
# Grouping data for EDA.
# Category for each item
item_category_mapping = items[['item_id','item_category_id']].drop_duplicates()
trans = pd.merge(trans, item_category_mapping, how='left', on='item_id')

# Sales by month
gp_month_mean = trans.groupby(['month'], as_index=False)['item_cnt_day'].mean()
gp_month_sum = trans.groupby(['month'], as_index=False)['item_cnt_day'].sum()

# Sales by item category
gp_category_mean = trans.groupby(['item_category_id'], as_index=False)['item_cnt_day'].mean()
gp_category_sum = trans.groupby(['item_category_id'], as_index=False)['item_cnt_day'].sum()

# Sales by shop
gp_shop_mean = trans.groupby(['shop_id'], as_index=False)['item_cnt_day'].mean()
gp_shop_sum = trans.groupby(['shop_id'], as_index=False)['item_cnt_day'].sum()


# f, axes = plt.subplots(2, 1, figsize=(22, 10), sharex=True)
# sns.barplot(x="item_category_id", y="item_cnt_day", data=gp_category_mean, ax=axes[0], palette="rocket").set_title("Monthly mean")
# sns.barplot(x="item_category_id", y="item_cnt_day", data=gp_category_sum, ax=axes[1], palette="rocket").set_title("Monthly sum")
# plt.show()
#
# f, axes = plt.subplots(2, 1, figsize=(22, 10), sharex=True)
# sns.lineplot(x="month", y="item_cnt_day", data=gp_month_mean, ax=axes[0], palette="rocket").set_title("Monthly mean")
# sns.lineplot(x="month", y="item_cnt_day", data=gp_month_sum, ax=axes[1], palette="rocket").set_title("Monthly sum")
# plt.show()
#
# f, axes = plt.subplots(2, 1, figsize=(22, 10), sharex=True)
# sns.barplot(x="shop_id", y="item_cnt_day", data=gp_shop_mean, ax=axes[0], palette="rocket").set_title("Monthly mean")
# sns.barplot(x="shop_id", y="item_cnt_day", data=gp_shop_sum, ax=axes[1], palette="rocket").set_title("Monthly sum")
# plt.show()

########################
# Feature Engineering
########################

# Sort by date
trans = trans.sort_values('date_block_num')

# Clip sales values into the [0, 20] range
trans['item_cnt_day'][trans['item_cnt_day'] < 0] = 0
trans['item_cnt_day'][trans['item_cnt_day'] > 20] = 20

# Drop item name column, consider it as un-useful for now
trans = trans.drop(columns = ['date'])

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
grid = pd.DataFrame(np.vstack(grid), columns = index_cols, dtype=np.int32)

# Aggregate to monthly data by month, shop, item
gb = trans.groupby(['date_block_num', 'year', 'month', 'shop_id', 'item_category_id', 'item_id']).agg({'item_cnt_day': 'sum', 'item_price': 'mean'}).reset_index()
gb.columns = ['date_block_num', 'year', 'month', 'shop_id', 'item_category_id', 'item_id', 'item_price', 'item_cnt_month']
train = pd.merge(grid, gb, how='left', on=['date_block_num', 'shop_id', 'item_id']).fillna(0)

train['target'] = train.sort_values('date_block_num').groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(-1)

del gb

train = downcast_dtypes(train)


#############################
# Adding lag based features
#############################

# Creating a lag term
cols_to_rename = ['item_cnt_month']

shift_range = [1, 2, 3, 6]

for month_shift in shift_range:
    train_shift = train[index_cols + cols_to_rename].copy()

    train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift

    foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x
    train_shift = train_shift.rename(columns=foo)
    train = pd.merge(train, train_shift, on=index_cols, how='left').fillna(0)

del train_shift

# Don't use old data from year 2013 because they don't have a lag
train = train[train['date_block_num'] >= 6]

# Category for each item
# item_category_mapping = items[['item_id','item_category_id']].drop_duplicates()
# item_price_mapping = trans[['item_id', 'item_price']].drop_duplicates()
#
# train = pd.merge(train, item_category_mapping, how='left', on='item_id')
# train = pd.merge(train, item_price_mapping, how='left', on='item_id')


###################
# Trend Features
###################
train['item_trend'] = train['item_cnt_month']

for lag in shift_range:
    ft_name = ('item_cnt_month_lag_%s' % lag)
    train['item_trend'] -= train[ft_name]

train['item_trend'] /= len(shift_range) + 1



###############################
# Rolling Window Calculations #
###############################
# Rolling Min
f_min = lambda x: x.rolling(window=3, min_periods=1).min()
# Rolling Max
f_max = lambda x: x.rolling(window=3, min_periods=1).max()
# Rolling Mean
f_mean = lambda x: x.rolling(window=3, min_periods=1).mean()
# Rolling Stardard Deviation
f_sd = lambda x: x.rolling(window=3, min_periods=1).std()

func_list = [f_min, f_max, f_mean, f_sd]
func_name = ['min', 'max', 'mean', 'sd']

for i in range(len(func_list)):
    train[('item_cnt_%s' % func_name[i])] = train.sort_values('date_block_num').groupby(['shop_id', 'item_category_id', 'item_id'])['item_cnt_month'].apply(func_list[i])

# Fill the empty std features with 0
train['item_cnt_sd'].fillna(0, inplace=True)


# List of all lagged features
# fit_cols = [col for col in train.columns if col[-1] in [str(item) for item in shift_range]]

# We will drop these at fitting stage
# to_drop_cols = list(set(list(train.columns)) - (set(fit_cols)|set(index_cols))) + ['date_block_num']



########################
# Group Based Features #
########################
# Item history min and max prices
gp_item_price = train.sort_values('date_block_num').groupby(['item_id'], as_index=False).agg({'item_price':[np.min, np.max]})
gp_item_price.columns = ['item_id', 'min_item_price', 'max_item_price']

train = pd.merge(train, gp_item_price, how='left', on='item_id')

# How much the price has changed since its historical high and low
train['price_increase'] = train['max_item_price'] - train['item_price']
train['price_decrease'] = train['item_price'] - train['min_item_price']

train.head().T

#####################
# Train/test split
#####################
train_set = train.query('date_block_num < 28').copy()
validation_set = train.query('date_block_num >= 28 and date_block_num < 33').copy()
test_set = train.query('date_block_num == 33').copy()

train_set.dropna(subset=['target'], inplace=True)
validation_set.dropna(subset=['target'], inplace=True)

train_set.dropna(inplace=True)
validation_set.dropna(inplace=True)

print('Train set records:', train_set.shape[0])
print('Validation set records:', validation_set.shape[0])
print('Test set records:', test_set.shape[0])

##########################
# Mean encoding features #
##########################
# To be done after train test split


# Shop mean sales
gp_shop = train_set.groupby('shop_id', as_index=False).agg({'target': ['mean', 'std']})
gp_shop.columns = ['shop_id', 'shop_mean', 'shop_std']

gp_item = train_set.groupby('item_id', as_index=False).agg({'target': ['mean', 'std']})
gp_item.columns = ['item_id', 'item_mean', 'item_std']

gp_shop_item = train_set.groupby(['shop_id', 'item_id'], as_index=False).agg({'target': ['mean', 'std']})
gp_shop_item.columns = ['shop_id', 'item_id', 'shop_item_mean', 'shop_item_std']

# Aggregate to item-month
gp_item_month = train_set.groupby(['date_block_num', 'item_id'], as_index=False).agg({'target': ['mean', 'std']})
gp_item_month.columns = ['date_block_num', 'item_id', 'item_month_mean', 'item_month_std']

# Aggregate to shop-month
gp_shop_month = train_set.groupby(['date_block_num', 'shop_id'], as_index=False).agg({'target': ['mean', 'std']})
gp_shop_month.columns = ['date_block_num', 'shop_id', 'shop_month_mean', 'shop_month_std']

# Join them to the training data
train_set = pd.merge(train_set, gp_shop, on='shop_id', how='left')
train_set = pd.merge(train_set, gp_item, on='item_id', how='left')
train_set = pd.merge(train_set, gp_shop_item, on=['shop_id', 'item_id'], how='left')
train_set = pd.merge(train_set, gp_item_month, on=['date_block_num', 'item_id'], how='left')
train_set = pd.merge(train_set, gp_shop_month, on=['date_block_num', 'shop_id'], how='left')

validation_set = pd.merge(validation_set, gp_shop, on='shop_id', how='left')
validation_set = pd.merge(validation_set, gp_item, on='item_id', how='left')
validation_set = pd.merge(validation_set, gp_shop_item, on=['shop_id', 'item_id'], how='left')
validation_set = pd.merge(validation_set, gp_item_month, on=['date_block_num', 'item_id'], how='left')
validation_set = pd.merge(validation_set, gp_shop_month, on=['date_block_num', 'shop_id'], how='left')

# Create train and validation sets and labels.
X_train = train_set.drop(['target', 'date_block_num'], axis=1)
Y_train = train_set['target'].astype(int)
X_validation = validation_set.drop(['target', 'date_block_num'], axis=1)
Y_validation = validation_set['target'].astype(int)

# Integer features (used by catboost model).
int_features = ['shop_id', 'item_id', 'year', 'month', 'item_category_id']

X_train[int_features] = X_train[int_features].astype('int32')
X_validation[int_features] = X_validation[int_features].astype('int32')

###############################
# Preparing the test dataset
###############################
# Keep the item shop combination where appeared last
latest_records = pd.concat([train_set, validation_set]).drop_duplicates(subset=['shop_id', 'item_id'], keep='last')
X_test = pd.merge(test, latest_records, on=['shop_id', 'item_id'], how='left', suffixes=['', '_'])
X_test['year'] = 2015
X_test['month'] = 9
X_test.drop('item_cnt_month', axis=1, inplace=True)
X_test[int_features] = X_test[int_features].astype('int32')
X_test = X_test[X_train.columns]

latest_shops = list(train['shop_id'].drop_duplicates())
print(len(latest_shops))
test_shops = list(test_shop_ids)
print(len(test_shops))

latest_items = list(train['item_id'].drop_duplicates())
print(len(latest_items))
test_item = list(test_item_ids)
print(len(test_item))

####################################
# A baseline model using catboost
####################################
pool = Pool(data=X_train, label=y_train)46
print(pool.get_feature_names())

cat_features = ['shop_id', 'item_id', 'item_category_id']
# Training 10 models with different random seed and average the score
scores = np.zeros(10)
for i in range(10):
    model = CatBoostRegressor(
        iterations=5,
        random_seed=i,
        learning_rate=0.1
    )
    model.fit(
        X_train, y_train,
        cat_features=cat_features,
        eval_set=(X_test, y_test)
    )
    print('Iteration' + str(i))
    scores[i] = model.best_score_['validation']['RMSE']

np.mean(scores)

pred = model.predict(data=X_test)

sns.jointplot(x=pred, y=y_test, height=8)
plt.show()


print('Model is fitted: ' + str(model.is_fitted()))
print('Model params:')
print(model.get_params())




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

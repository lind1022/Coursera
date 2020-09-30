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
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic

from catboost import *
import catboost
from catboost import Pool
from catboost import CatBoostClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance
%matplotlib qt

hha

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

# Sort by date
trans = trans.sort_values('date_block_num')

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
block_nums = trans['date_block_num'].unique()
all_shops = trans.loc[:,'shop_id'].unique()
all_items = trans.loc[:,'item_id'].unique()
grid = np.array(list(product(*[test_shop_ids, test_item_ids, block_nums])),dtype='int32')

# Turn the grid into a dataframe
grid = pd.DataFrame(np.vstack(grid), columns = index_cols, dtype=np.int32)

# need to add category, year and month mapping
grid = pd.merge(grid, item_category_mapping, on='item_id', how='left')
grid = pd.merge(grid, shops, on='shop_id', how='left')
grid = pd.merge(grid, item_categories, on='item_category_id', how='left')

# Number of days in the month
days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31], index=range(1, 13))
year_date_mapping = trans[['date_block_num', 'year', 'month']].drop_duplicates()

year_date_mapping['days'] = year_date_mapping['month'].map(days)

grid = pd.merge(grid, year_date_mapping, on='date_block_num', how='left')

# Aggregate to monthly data by month, shop, item
col_order = ['date_block_num', 'shop_id', 'item_id', 'item_cnt_day', 'item_price']
gb = trans.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False).agg({'item_cnt_day': 'sum', 'item_price': 'mean'})[col_order]
gb.columns = ['date_block_num', 'shop_id', 'item_id', 'item_cnt_month', 'item_price']

plt.subplots(figsize=(22, 8))
sns.boxplot(gb['item_cnt_month'])
plt.show()

# Clip sales values into the [0, 20] range
gb['item_cnt_month'][gb['item_cnt_month'] < 0] = 0
gb['item_cnt_month'][gb['item_cnt_month'] > 20] = 20

train = pd.merge(grid, gb, how='left', on=['date_block_num', 'shop_id', 'item_id']).fillna(0)
del gb

# Create target variable
train['target'] = train.sort_values('date_block_num').groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(-1)

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
gp_item_month = train_set.groupby(['month', 'item_id'], as_index=False).agg({'target': ['mean', 'std']})
gp_item_month.columns = ['month', 'item_id', 'item_month_mean', 'item_month_std']

# Aggregate to shop-month
gp_shop_month = train_set.groupby(['month', 'shop_id'], as_index=False).agg({'target': ['mean', 'std']})
gp_shop_month.columns = ['month', 'shop_id', 'shop_month_mean', 'shop_month_std']

# Year mean encoding.
gp_year = train_set.groupby(['year'], as_index=False).agg({'target': ['mean']})
gp_year.columns = ['year', 'year_mean']

# Month mean encoding.
gp_month = train_set.groupby(['month'], as_index=False).agg({'target': ['mean']})
gp_month.columns = ['month', 'month_mean']

# Category mean encoding
gp_category = train_set.groupby(['item_category_id'], as_index=False).agg({'target': ['mean']})
gp_category.columns = ['item_category_id', 'category_mean']

# Month Category mean encoding
gp_category_month = train_set.groupby(['month', 'item_category_id'], as_index=False).agg({'target': ['mean']})
gp_category_month.columns = ['month', 'item_category_id', 'category_month_mean']

# Category shop mean encoding
gp_month_category_shop = train_set.groupby(['month', 'item_category_id', 'shop_id'], as_index=False).agg({'target': ['mean']})
gp_month_category_shop.columns = ['month', 'item_category_id', 'shop_id', 'month_category_shop_mean']

# Shop type mean encoding
gp_shop_type = train_set.groupby(['shop_id', 'type_code'], as_index=False).agg({'target': ['mean']})
gp_shop_type.columns = ['shop_id', 'type_code', 'shop_type_mean']

# Month shop type mean encoding
gp_month_shop_type = train_set.groupby(['month', 'shop_id', 'type_code'], as_index=False).agg({'target': ['mean']})
gp_month_shop_type.columns = ['month', 'shop_id', 'type_code', 'month_shop_type_mean']

# Shop subtype mean encoding
gp_month_shop_subtype = train_set.groupby(['month', 'shop_id', 'subtype_code'], as_index=False).agg({'target': ['mean']})
gp_month_shop_subtype.columns = ['month', 'shop_id', 'subtype_code', 'month_shop_subtype_mean']

# Month city mean encoding
gp_month_city = train_set.groupby(['month', 'city_code'], as_index=False).agg({'target': ['mean']})
gp_month_city.columns = ['month', 'city_code', 'month_city_mean']

# Month City Category Encoding
gp_month_city_category = train_set.groupby(['month', 'city_code', 'item_category_id'], as_index=False).agg({'target': ['mean']})
gp_month_city_category.columns = ['month', 'city_code', 'item_category_id', 'month_city_category_mean']



# Join them to the training data
train_set = pd.merge(train_set, gp_shop, on='shop_id', how='left')
train_set = pd.merge(train_set, gp_item, on='item_id', how='left')
train_set = pd.merge(train_set, gp_shop_item, on=['shop_id', 'item_id'], how='left')
train_set = pd.merge(train_set, gp_item_month, on=['month', 'item_id'], how='left')
train_set = pd.merge(train_set, gp_shop_month, on=['month', 'shop_id'], how='left')
train_set = pd.merge(train_set, gp_year, on='year', how='left')
train_set = pd.merge(train_set, gp_month, on='month', how='left')
train_set = pd.merge(train_set, gp_category, on='item_category_id', how='left')
train_set = pd.merge(train_set, gp_category_month, on=['month', 'item_category_id'], how='left')

train_set = pd.merge(train_set, gp_month_category_shop, on=['month', 'item_category_id', 'shop_id'], how='left')
train_set = pd.merge(train_set, gp_shop_type, on=['shop_id', 'type_code'], how='left')
train_set = pd.merge(train_set, gp_month_shop_type, on=['month', 'shop_id', 'type_code'], how='left')
train_set = pd.merge(train_set, gp_month_shop_subtype, on=['month', 'shop_id', 'subtype_code'], how='left')
train_set = pd.merge(train_set, gp_month_city, on=['month', 'city_code'], how='left')
train_set = pd.merge(train_set, gp_month_city_category, on=['month', 'city_code', 'item_category_id'], how='left')


validation_set = pd.merge(validation_set, gp_shop, on='shop_id', how='left')
validation_set = pd.merge(validation_set, gp_item, on='item_id', how='left')
validation_set = pd.merge(validation_set, gp_shop_item, on=['shop_id', 'item_id'], how='left')
validation_set = pd.merge(validation_set, gp_item_month, on=['month', 'item_id'], how='left')
validation_set = pd.merge(validation_set, gp_shop_month, on=['month', 'shop_id'], how='left')
validation_set = pd.merge(validation_set, gp_year, on='year', how='left')
validation_set = pd.merge(validation_set, gp_month, on='month', how='left')
validation_set = pd.merge(validation_set, gp_category, on='item_category_id', how='left')
validation_set = pd.merge(validation_set, gp_category_month, on=['month', 'item_category_id'], how='left')

validation_set = pd.merge(validation_set, gp_month_category_shop, on=['month', 'item_category_id', 'shop_id'], how='left')
validation_set = pd.merge(validation_set, gp_shop_type, on=['shop_id', 'type_code'], how='left')
validation_set = pd.merge(validation_set, gp_month_shop_type, on=['month', 'shop_id', 'type_code'], how='left')
validation_set = pd.merge(validation_set, gp_month_shop_subtype, on=['month', 'shop_id', 'subtype_code'], how='left')
validation_set = pd.merge(validation_set, gp_month_city, on=['month', 'city_code'], how='left')
validation_set = pd.merge(validation_set, gp_month_city_category, on=['month', 'city_code', 'item_category_id'], how='left')

# Create train and validation sets and labels.
X_train = train_set.drop(['target', 'date_block_num'], axis=1)
Y_train = train_set['target'].astype(int)
X_validation = validation_set.drop(['target', 'date_block_num'], axis=1)
Y_validation = validation_set['target'].astype(int)

# Integer features (used by catboost model).
int_features = ['shop_id', 'item_id', 'year', 'month', 'item_category_id', 'city_code', 'type_code', 'subtype_code']
#
# X_train[int_features] = X_train[int_features].astype('int32')
# X_validation[int_features] = X_validation[int_features].astype('int32')

###############################
# Preparing the test dataset
###############################
# Keep the item shop combination where appeared last
latest_records = pd.concat([train_set, validation_set]).drop_duplicates(subset=['shop_id', 'item_id'], keep='last')
X_test = pd.merge(test, latest_records, on=['shop_id', 'item_id'], how='left', suffixes=['', '_'])
X_test['year'] = 2015
X_test['month'] = 9
X_test.drop('target', axis=1, inplace=True)
X_test[int_features] = X_test[int_features].astype('int32')
X_test = X_test[X_train.columns]

# Replace missing values with the median of each shop.
# sets = [X_train, X_validation, X_test]
# for dataset in sets:
#     for shop_id in dataset['shop_id'].unique():
#         for column in dataset.columns:
#             shop_median = dataset[(dataset['shop_id'] == shop_id)][column].median()
#             dataset.loc[(dataset[column].isnull()) & (dataset['shop_id'] == shop_id), column] = shop_median


X_train[int_features] = X_train[int_features].astype('int32')
X_validation[int_features] = X_validation[int_features].astype('int32')


##########
# XGboost
##########
# Use only part of features on XGBoost.
xgb_features = ['item_cnt','item_cnt_mean', 'item_cnt_std', 'item_cnt_shifted1',
                'item_cnt_shifted2', 'item_cnt_shifted3', 'shop_mean',
                'shop_item_mean', 'item_trend', 'mean_item_cnt']
xgb_train = X_train[xgb_features]
xgb_val = X_validation[xgb_features]
xgb_test = X_test[xgb_features]

model = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.3,
    seed=42)

model.fit(
    X_train,
    Y_train,
    eval_metric="rmse",
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
    verbose=True,
    early_stopping_rounds = 10)





####################################
# A baseline model using catboost
####################################
pool = Pool(data=X_train, label=Y_train)
print(pool.get_feature_names())

catboost_model = CatBoostRegressor(
    iterations=500,
    max_ctr_complexity=4,
    random_seed=0,
    od_type='Iter',
    od_wait=25,
    verbose=50,
    depth=4
)

catboost_model.fit(
    X_train, Y_train,
    cat_features=int_features,
    eval_set=(X_validation, Y_validation)
)

# catboost_model = CatBoostRegressor(
#     iterations=5,
#     random_seed=3,
#     learning_rate=0.1
# )
# catboost_model.fit(
#     X_train, Y_train,
#     cat_features=cat_features,
#     eval_set=(X_validation, Y_validation)
# )
catboost_model.best_score_['validation']['RMSE']


catboost_train_pred = catboost_model.predict(X_train)
catboost_val_pred = catboost_model.predict(X_validation)
catboost_test_pred = catboost_model.predict(X_test)

def model_performance_sc_plot(predictions, labels, title):
    # Get min and max values of the predictions and labels.
    min_val = max(max(predictions), max(labels))
    max_val = min(min(predictions), min(labels))
    # Create dataframe with predicitons and labels.
    performance_df = pd.DataFrame({"Label":labels})
    performance_df["Prediction"] = predictions
    # Plot data
    sns.jointplot(y="Label", x="Prediction", data=performance_df, kind="reg", height=7)
    plt.plot([min_val, max_val], [min_val, max_val], 'm--')
    plt.title(title, fontsize=9)
    plt.show()

# model_performance_sc_plot(catboost_train_pred, Y_train, 'Train')
model_performance_sc_plot(catboost_val_pred, Y_validation, 'Validation')

feature_score = pd.DataFrame(list(zip(X_train.dtypes.index, catboost_model.get_feature_importance(Pool(X_train, label=Y_train, cat_features=cat_features)))), columns=['Feature','Score'])
feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')

plt.rcParams["figure.figsize"] = (19, 6)
ax = feature_score.plot('Feature', 'Score', kind='bar', color='c')
ax.set_title("Catboost Feature Importance Ranking", fontsize = 14)
ax.set_xlabel('')
rects = ax.patches
labels = feature_score['Score'].round(2)

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 0.45, label, ha='center', va='bottom')

plt.show()


###########################################################
# Creating the prediction baseline with Oct 15 sales
###########################################################

pred = pd.DataFrame(catboost_test_pred, columns=['item_cnt_month'])
test['item_cnt_month'] = pred.clip(0, 20)

# File to submit
submit = test[['ID', 'item_cnt_month']]
submit.to_csv("submission.csv", index = False)


# end of script

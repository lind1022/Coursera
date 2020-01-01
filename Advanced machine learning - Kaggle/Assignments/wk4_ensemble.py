import numpy as np
import pandas as pd
import sklearn
import scipy.sparse
import lightgbm as lgb

for p in [np, pd, scipy, sklearn, lgb]:
    print (p.__name__, p.__version__)

import gc
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from itertools import product

# Data load
DATA_FOLDER = 'C:/Lin/Data science/Github repo/Coursera/Advanced machine learning - Kaggle/Final project'
sales           = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv.gz'))
items           = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))
item_cats = pd.read_csv(os.path.join(DATA_FOLDER, 'item_categories.csv'))
shops           = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))

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


# Get a feature matrix
# Create "grid" with columns
index_cols = ['shop_id', 'item_id', 'date_block_num']

# For every month we create a grid from all shops/items combinations from that month
grid = []
for block_num in sales['date_block_num'].unique():
    cur_shops = sales.loc[sales['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = sales.loc[sales['date_block_num'] == block_num, 'item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

# Turn the grid into a dataframe
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)

# Groupby data to get shop-item-month aggregates
gb = sales.groupby(index_cols,as_index=False).agg({'item_cnt_day':{'target':'sum'}})
# Fix column names
gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]
# Join it to the grid
all_data = pd.merge(grid, gb, how='left', on=index_cols).fillna(0)

# Same as above but with shop-month aggregates
gb = sales.groupby(['shop_id', 'date_block_num'],as_index=False).agg({'item_cnt_day':{'target_shop':'sum'}})
gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]
all_data = pd.merge(all_data, gb, how='left', on=['shop_id', 'date_block_num']).fillna(0)

# Same as above but with item-month aggregates
gb = sales.groupby(['item_id', 'date_block_num'],as_index=False).agg({'item_cnt_day':{'target_item':'sum'}})
gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]
all_data = pd.merge(all_data, gb, how='left', on=['item_id', 'date_block_num']).fillna(0)

# Downcast dtypes from 64 to 32 bit to save memory
all_data = downcast_dtypes(all_data)
del grid, gb
gc.collect();

# List of columns that we will use to create lags
cols_to_rename = list(all_data.columns.difference(index_cols))

shift_range = [1, 2, 3, 4, 5, 12]

for month_shift in shift_range:
    train_shift = all_data[index_cols + cols_to_rename].copy()

    train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift

    foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x
    train_shift = train_shift.rename(columns=foo)

    all_data = pd.merge(all_data, train_shift, on=index_cols, how='left').fillna(0)

del train_shift

# Don't use old data from year 2013
all_data = all_data[all_data['date_block_num'] >= 12]

# List of all lagged features
fit_cols = [col for col in all_data.columns if col[-1] in [str(item) for item in shift_range]]
# We will drop these at fitting stage
to_drop_cols = list(set(list(all_data.columns)) - (set(fit_cols)|set(index_cols))) + ['date_block_num']

# Category for each item
item_category_mapping = items[['item_id','item_category_id']].drop_duplicates()

all_data = pd.merge(all_data, item_category_mapping, how='left', on='item_id')
all_data = downcast_dtypes(all_data)
gc.collect();



#####################
# Train/test split
#####################
# Save `date_block_num`, as we can't use them as features, but will need them to split the dataset into parts
dates = all_data['date_block_num']

last_block = dates.max()
print('Test `date_block_num` is %d' % last_block)

dates_train = dates[dates <  last_block]
dates_test  = dates[dates == last_block]

X_train = all_data.loc[dates <  last_block].drop(to_drop_cols, axis=1)
X_test =  all_data.loc[dates == last_block].drop(to_drop_cols, axis=1)

y_train = all_data.loc[dates <  last_block, 'target'].values
y_test =  all_data.loc[dates == last_block, 'target'].values


#############################
# Test meta features
#############################
lr = LinearRegression()
lr.fit(X_train.values, y_train)
pred_lr = lr.predict(X_test.values)

print('Test R-squared for linreg is %f' % r2_score(y_test, pred_lr))

lgb_params = {
               'feature_fraction': 0.75,
               'metric': 'rmse',
               'nthread':1,
               'min_data_in_leaf': 2**7,
               'bagging_fraction': 0.75,
               'learning_rate': 0.03,
               'objective': 'mse',
               'bagging_seed': 2**7,
               'num_leaves': 2**7,
               'bagging_freq':1,
               'verbose':0
              }

model = lgb.train(lgb_params, lgb.Dataset(X_train, label=y_train), 100)
pred_lgb = model.predict(X_test)

print('Test R-squared for LightGBM is %f' % r2_score(y_test, pred_lgb))

#############################
# Train meta features
#############################
'''
Here, we will use duration T equal to month and M=15.

That is, you need to get predictions (meta-features) from linear regression and LightGBM
 for months 27, 28, 29, 30, 31, 32. Use the same parameters as in above models.
'''

dates_train_level2 = dates_train[dates_train.isin([27, 28, 29, 30, 31, 32])]

# That is how we get target for the 2nd level dataset
y_train_level2 = y_train[dates_train.isin([27, 28, 29, 30, 31, 32])]

all_data.groupby('date_block_num').count()

# And here we create 2nd level feeature matrix, init it with zeros first
X_train_level2 = np.zeros([y_train_level2.shape[0], 2])
row_start = 0
# Now fill `X_train_level2` with metafeatures
for cur_block_num in [27, 28, 29, 30, 31, 32]:

    print(cur_block_num)

    '''
        1. Split `X_train` into parts
           Remember, that corresponding dates are stored in `dates_train`
        2. Fit linear regression
        3. Fit LightGBM and put predictions
        4. Store predictions from 2. and 3. in the right place of `X_train_level2`.
           You can use `dates_train_level2` for it
           Make sure the order of the meta-features is the same as in `X_test_level2`
    '''

    #  YOUR CODE GOES HERE
    # Separate the train data
    X_train_subset = X_train[dates_train.isin(list(range(12, cur_block_num)))]
    y_train_subset = y_train[dates_train.isin(list(range(12, cur_block_num)))]
    lr.fit(X_train_subset.values, y_train_subset)
    model = lgb.train(lgb_params, lgb.Dataset(X_train_subset, label=y_train_subset), 100)
    # Separate the testing data
    pred_X = X_train[dates_train.isin([cur_block_num])]

    row_end = row_start + pred_X.shape[0]
    index = list(range(row_start, row_end))

    X_train_level2[index,0] = lr.predict(pred_X.values)
    X_train_level2[index,1] = model.predict(pred_X)

    row_start = row_end

# Sanity check
assert np.all(np.isclose(X_train_level2.mean(axis=0), [ 1.50148988,  1.38811989]))

# Testing panel - not part of assignment
cur_block_num = 27
# Separate the data
X_train_level2 = np.zeros([y_train_level2.shape[0], 2])
y_train_subset = y_train[dates_train.isin(list(range(12, cur_block_num)))]
X_train_subset = X_train[dates_train.isin(list(range(12, cur_block_num)))]
# Train the models
lr.fit(X_train_subset.values, y_train_subset)
model = lgb.train(lgb_params, lgb.Dataset(X_train_subset, label=y_train_subset), 100)
# Separate the testing data
pred_X = X_train[dates_train.isin([cur_block_num])]

row_end = pred_X.shape[0]
index = list(range(0, row_end))

X_train_level2[list(range(0,row_end)),0] = lr.predict(pred_X.values)
X_train_level2[list(range(0,row_end)),1] = model.predict(pred_X)

# YOUR CODE GOES HERE
plt.scatter(X_train_level2[:,0], X_train_level2[:,1])

###############
# Ensembling
###############
'''
Start with simple linear convex mix:
mix = alpha * linreg_pred + (1 - alpha) * lgb_pred
'''
alpha = 0.05
mix = alpha * X_train_level2[:,0] + (1 - alpha) * X_train_level2[:,1]
r2_score(y_train_level2, mix)

alphas_to_try = np.linspace(0, 1, 1001)
# YOUR CODE GOES HERE
r2 = 0
for alpha in alphas_to_try:
    mix = alpha * X_train_level2[:,0] + (1 - alpha) * X_train_level2[:,1]
    r2_new = r2_score(y_train_level2, mix)
    if r2_new > r2:
        best_alpha = alpha
    r2 = r2_new
print(best_alpha)


# best_alpha = best_alpha # YOUR CODE GOES HERE
mix = best_alpha * X_train_level2[:,0] + (1 - best_alpha) * X_train_level2[:,1]
r2_train_simple_mix = r2_score(y_train_level2, mix)

print('Best alpha: %f; Corresponding r2 score on train: %f' % (best_alpha, r2_train_simple_mix))

test_preds = best_alpha * X_test_level2[:,0] + (1 - best_alpha) * X_test_level2[:,1]# YOUR CODE GOES HERE
r2_test_simple_mix = r2_score(y_test, test_preds)# YOUR CODE GOES HERE

print('Test R-squared for simple mix is %f' % r2_test_simple_mix)

############
# Stacking
############

'''
Now, we will try a more advanced ensembling technique.
Fit a linear regression model to the meta-features. Use the same parameters as in the model above.
'''
# YOUR CODE GOES HERE
lr.fit(X_train_level2, y_train_level2)

train_preds = lr.predict(X_train_level2)# YOUR CODE GOES HERE
r2_train_stacking = r2_score(y_train_level2, train_preds)# YOUR CODE GOES HERE

test_preds = lr.predict(X_test_level2)# YOUR CODE GOES HERE
r2_test_stacking = r2_score(y_test, test_preds)# YOUR CODE GOES HERE

print('Train R-squared for stacking is %f' % r2_train_stacking)
print('Test  R-squared for stacking is %f' % r2_test_stacking)








# end of script

import pandas as pd
import numpy as np
from itertools import product
import os
from sklearn.model_selection import KFold


# DATA_FOLDER = 'C:/Users/lind/Coursera/Advanced machine learning - Kaggle/Final project'

DATA_FOLDER = 'C:/Lin/Data science/Github repo/Coursera/Advanced machine learning - Kaggle/Final project'
sales           = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv.gz'))

index_cols = ['shop_id', 'item_id', 'date_block_num']

# For every month we create a grid from all shops/items combinations from that month
grid = []
for block_num in sales['date_block_num'].unique():
    cur_shops = sales[sales['date_block_num']==block_num]['shop_id'].unique()
    cur_items = sales[sales['date_block_num']==block_num]['item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

#turn the grid into pandas dataframe
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)

#get aggregated values for (shop_id, item_id, month)
gb = sales.groupby(index_cols,as_index=False).agg({'item_cnt_day':{'target':'sum'}})

#fix column names
gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]
#join aggregated data to the grid
all_data = pd.merge(grid, gb, how='left', on=index_cols).fillna(0)
#sort the data
all_data.sort_values(['date_block_num','shop_id','item_id'], inplace=True)

#############################################
# Mean encoding without regularisation
#############################################

# Method 1
# Calculate a mapping: {item_id: target_mean}
item_id_target_mean = all_data.groupby('item_id').target.mean()

# In our non-regularized case we just *map* the computed means to the `item_id`'s
all_data['item_target_enc'] = all_data['item_id'].map(item_id_target_mean)

# Fill NaNs
all_data['item_target_enc'].fillna(0.3343, inplace=True)

# Print correlation
encoded_feature = all_data['item_target_enc'].values
print(np.corrcoef(all_data['target'].values, encoded_feature)[0][1])


# Method 2
'''
     Differently to `.target.mean()` function `transform`
   will return a dataframe with an index like in `all_data`.
   Basically this single line of code is equivalent to the first two lines from of Method 1.
'''
all_data['item_target_enc'] = all_data.groupby('item_id')['target'].transform('mean')

# Fill NaNs
all_data['item_target_enc'].fillna(0.3343, inplace=True)

# Print correlation
encoded_feature = all_data['item_target_enc'].values
print(np.corrcoef(all_data['target'].values, encoded_feature)[0][1])


######################
# 1. K-folds scheme
######################

'''
First, implement KFold scheme with five folds. Use KFold(5) from sklearn.model_selection.

Split your data in 5 folds with sklearn.model_selection.KFold with shuffle=False argument.
Iterate through folds: use all but the current fold to calculate mean target for each level item_id, and fill the current fold.

See the Method 1 from the example implementation. In particular learn what map and pd.Series.map functions do. They are pretty handy in many situations.
'''


# YOUR CODE GOES HERE
kf = KFold(n_splits=5, shuffle=False)
# item_id_target_mean = pd.Series('NaN', index = sales['item_id'].sort_values().unique())
for train_index, test_index in kf.split(all_data):
    item_id_target_mean = all_data.loc[train_index].groupby('item_id').target.mean()
    all_data.loc[test_index, 'item_target_enc'] = all_data['item_id'].loc[test_index].map(item_id_target_mean)

all_data['item_target_enc'].fillna(0.3343, inplace=True)
encoded_feature = all_data['item_target_enc'].values

# You will need to compute correlation like that
corr = np.corrcoef(all_data['target'].values, encoded_feature)[0][1]
print(corr)


############################
# 2. Leave-one-out scheme
############################
'''
To implement a faster version, note, that to calculate mean target value using all the objects but one given object, you can:
    1. Calculate sum of the target values using all the objects.
    2. Then subtract the target of the given object and divide the resulting value by n_objects - 1.

Note that you do not need to perform 1. for every object. And 2. can be implemented without any for loop.
It is the most convenient to use .transform function as in Method 2.
'''
# YOUR CODE GOES HERE
all_data['item_id_sum'] = all_data.groupby('item_id')['target'].transform('sum')
all_data['n_objects'] = all_data.groupby('item_id')['target'].transform('count')
all_data['item_target_enc'] = (all_data.item_id_sum - all_data.target) / (all_data.n_objects - 1)

all_data['item_target_enc'].fillna(0.3343, inplace=True)
encoded_feature = all_data['item_target_enc'].values

corr = np.corrcoef(all_data['target'].values, encoded_feature)[0][1]
print(corr)


#################
# 3. Smoothing
#################

'''
Next, implement smoothing scheme with alpha=100. Use the formula from the first slide in the video and 0.33430.3343 as globalmean.
Note that nrows is the number of objects that belong to a certain category (not the number of rows in the dataset).
'''
# YOUR CODE GOES HERE
global_mean = all_data['target'].mean()
alpha = 100

all_data['mean_target'] = all_data.groupby('item_id')['target'].transform('mean')
all_data['nrows'] = all_data.groupby('item_id')['target'].transform('count')

all_data['item_target_enc'] = (all_data.mean_target * all_data.nrows + global_mean * alpha) / (all_data.nrows + alpha)

encoded_feature = all_data['item_target_enc'].values
corr = np.corrcoef(all_data['target'].values, encoded_feature)[0][1]
print(corr)

##############################
# 4. Expanding mean scheme
##############################

'''
Finally, implement the expanding mean scheme. It is basically already implemented
for you in the video, but you can challenge yourself and try to implement it yourself.
You will need cumsum and cumcount functions from pandas.
'''




















# end of script

import pandas as pd
import numpy as np
np.set_printoptions(precision=4)
import catboost
from catboost import datasets
from catboost import *

train_df, test_df = catboost.datasets.amazon()
train_df.head()


# Label values extraction
y = train_df.ACTION
X = train_df.drop('ACTION', axis=1)

# Categorical features declaration
cat_features = list(range(0, X.shape[1]))
print(cat_features)

'''
Now it makes sense to ananyze the dataset.
First you need to calculate how many positive and negative objects are present in the train dataset.
'''

# Question 1 & 2 - how many negative/positive objects are present in the train dataset X?
y.value_counts()

# Question 3 - How many unique values has feature RESOURCE?
X.RESOURCE.unique().shape

'''
Now we can create a Pool object. This type is used for datasets in CatBoost.
You can also use numpy array or dataframe. Working with Pool class is the most
efficient way in terms of memory and speed. We recommend to create Pool from
file in case if you have your data on disk or from FeaturesData if you use numpy.
'''
import numpy as np
from catboost import Pool

pool1 = Pool(data=X, label=y, cat_features=cat_features)
# pool2 = Pool(data='C:/Users/lukei/Anaconda2/envs/linspython3/Lib/site-packages/catboost/cached_datasets/amazon/train.csv', delimiter=',', has_header=True)
# pool3 = Pool(data=X, cat_features=cat_features)

print('Dataset shape')
print('dataset 1:' + str(pool1.shape))

print('\n')
print('Column names')
print('dataset 1: ')
print(pool1.get_feature_names())
# print('\ndataset 2:')
# print(pool2.get_feature_names())
# print('\ndataset 3:')
# print(pool3.get_feature_names())


'''
Split data into train and validation
When you will be training your model, you will have to detect overfitting and
select best parameters. To do that you need to have a validation dataset. Normally
you would be using some random split, for example train_test_split from
sklearn.model_selection.
But for the purpose of this homework the train part will be the first 80% of the
data and the evaluation part will be the last 20% of the data.
'''
train_count = int(X.shape[0] * 0.8)

X_train = X.iloc[:train_count,:]
y_train = y[:train_count]
X_validation = X.iloc[train_count:, :]
y_validation = y[train_count:]


#############
# Training
#############

# Now we train our first model
from catboost import CatBoostClassifier
model = CatBoostClassifier(
    iterations=5,
    random_seed=0,
    learning_rate=0.1
)
model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_validation, y_validation),
    logging_level='Silent'
)
print('Model is fitted: ' + str(model.is_fitted()))
print('Model params:')
print(model.get_params())


'''
You can see in stdout values of the loss function on each iteration, or on each
 k-th iteration. You can also see how much time passed since the start of the
 training and how much time is left.
'''

#################
# Random Seed
#################

'''
If you don't specify random_seed then random seed will be set to a new value each
time. After the training has finished you can look on the value of the random
seed that was set. If you train again with this random_seed, you will get the
same results.
'''
from catboost import CatBoostClassifier
model = CatBoostClassifier(
    iterations=5
)
model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_validation, y_validation),
)

random_seed = model.random_seed_
print('Used random seed = ' + str(random_seed))
model = CatBoostClassifier(
    iterations=5,
    random_seed=random_seed
)
model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_validation, y_validation),
)

'''
Try training 10 models with parameters and calculate mean and the standart
deviation of Logloss error on validation dataset.
'''

# Question 4
'''
What is the mean value of the Logloss metric on validation dataset
(X_validation, y_validation) after 10 times training `CatBoostClassifier` with
different random seeds in the following way:
'''
model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.1,
    random_seed={my_random_seed}
)
model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_validation, y_validation),
)












# end of script

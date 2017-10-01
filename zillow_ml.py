import numpy as np
import gc

import pandas as pd
#import xgboost as xgb
from catboost import CatBoostRegressor

from data_reducing import Reducer

print('Loading data ...')

train = pd.read_csv('data/train_2016_v2.csv', parse_dates=['transactiondate'])
prop = pd.read_csv('data/properties_2016.csv')
sample = pd.read_csv('data/sample_submission.csv')


# todo: Feature Engineering
train['transaction_month'] = train.transactiondate.dt.month.astype(np.uint8)
train['transaction_day'] = train.transactiondate.dt.day.astype(np.uint8)
train['transactiion_year'] = train.transactiondate.dt.year.astype(np.uint16)


print('Reduce data types..')
reducer = Reducer()
prop = reducer.reduce(prop, verbose=False)

df_train = train.merge(prop, how='left', on='parcelid')

cols_to_drop = ['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc',
                'propertycountylandusecode']
x_train = df_train.drop(cols_to_drop, axis=1)
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = x_train.loc[:, c].notnull()

del df_train; gc.collect()

# todo: modeling

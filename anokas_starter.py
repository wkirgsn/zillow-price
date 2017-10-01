import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
import gc

print('Loading data ...')

train = pd.read_csv('data/train_2016_v2.csv', parse_dates=['transactiondate'])
prop = pd.read_csv('data/properties_2016.csv')
sample = pd.read_csv('data/sample_submission.csv')

print('Binding to lower types')

for c, dtype in zip(prop.columns, prop.dtypes):
    if dtype == np.float64:
        prop[c] = prop[c].astype(np.float32)
    elif dtype == np.int64:
        prop[c] = prop[c].astype(np.int32)

print('Creating training set ...')

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

cat = CatBoostRegressor()
cat.fit(x_train, y_train)

split = 80000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], \
                                     x_train[split:], y_train[split:]

print('Building DMatrix...')

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

del x_train, x_valid; gc.collect()

print('Training ...')

params = {}
params['eta'] = 0.02
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['max_depth'] = 4
params['silent'] = 1


watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100,
                verbose_eval=10)

del d_train, d_valid

print('Building test set ...')

sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')

del prop; gc.collect()

x_test = df_test.loc[:, train_columns]

nancols = x_test.dtypes[x_test.dtypes == object].index.values.tolist()
x_test[nancols] = x_test.loc[:, nancols].notnull()

del df_test, sample; gc.collect()

p_test_cat = cat.predict(x_test)
d_test = xgb.DMatrix(x_test)
del x_test; gc.collect()

print('Predicting on test ...')

p_test = clf.predict(d_test)

del d_test; gc.collect()

sub = pd.read_csv('data/sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test_cat

print('Writing csv ...')
sub.to_csv('out/xgb_starter_compressed_cat.csv.gz', index=False,
           float_format='%.4g',
           compression='gzip')

print('ensemble..')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = (p_test_cat + p_test)/2

print('Writing csv ...')
sub.to_csv('out/xgb_starter_compressed_catxgb_mean.csv.gz', index=False,
           float_format='%.4g',
           compression='gzip')

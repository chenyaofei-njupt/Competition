# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:19:49 2017

@author: loveya
"""
import lightgbm as lgb
import pandas as pd
import numpy as np
from pandas import DataFrame

df_train = pd.read_csv('data_train_selected_df')
X_selected=pd.read_csv('X_selected.csv')
df_train.columns=X_selected.columns
X_train=np.array(df_train)
y_train = pd.read_csv('label_clean.csv')
y_train=y_train.iloc[:,0].values
data_test=pd.read_csv('/data/第1题：算法题数据/数据集1_用户标签_本地_测试集.csv')
data_test['是否有出境行为']=np.where(data_test['是否有出境行为']=='是',1,0)
data_test['是否有跨省行为']=np.where(data_test['是否有跨省行为']=='是',1,0)
data_test=data_test.drop(['手机品牌', '手机终端型号', '漫入省份', '漫出省份'],axis=1)
data_test_selected=data_test[X_selected.columns]
X_test=np.array(data_test_selected)
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'num_leaves': 31,
    'learning_rate': 0.03,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
print('Start training...')
# train
gbm = lgb.train(params,lgb_train,num_boost_round=120, early_stopping_rounds=5)
print('Save model...')
# save model to file
gbm.save_model('model.txt')
print('Start predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
a=data_test.iloc[:,0]
label_pred1=DataFrame(y_pred)
df=pd.concat([a,label_pred1],axis=1)
df.to_csv('result.csv',index=False,header=['IMEI','SCORE'],float_format='%.5f')
print('Feature names:', gbm.feature_name())
print('Calculate feature importances...')
# feature importances
print('Feature importances:', list(gbm.feature_importance()))
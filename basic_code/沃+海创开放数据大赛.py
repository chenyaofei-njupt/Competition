# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 20:27:38 2017

@author: loveya
"""
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pandas import DataFrame
"""读取数据和数据清洗"""
data_train=pd.read_csv('/data/第1题：算法题数据/数据集1_用户标签_本地_训练集.csv')
label_train=pd.read_csv('/data/第1题：算法题数据/数据集2_用户是否去过迪士尼_训练集.csv')
data_train['是否有出境行为']=np.where(data_train['是否有出境行为']=='是',1,0)
data_train['是否有跨省行为']=np.where(data_train['是否有跨省行为']=='是',1,0)
data_train=data_train.drop(['手机品牌', '手机终端型号', '漫入省份', '漫出省份'],axis=1)
label_train=label_train['是否去过迪士尼'] 
data=pd.concat([data_train,label_train],axis=1)
data_cleaned=data.dropna()
data.isnull().sum()
data_train_clean=data_cleaned.iloc[:,1:339].values
label_train_clean=data_cleaned.iloc[:,339].values
label_clean=DataFrame(label_train_clean)
label_clean.to_csv('label_clean.csv',index=False)

#np.any(np.isnan(data_train_clean))
#np.isnan(data_train_clean).sum()

#通过随机森林判定特征的重要性 
feat_labels=data_cleaned.columns[1:339]
forest=RandomForestClassifier(n_estimators=100,random_state=0,n_jobs=-1)
forest.fit(data_train_clean,label_train_clean)
importances=forest.feature_importances_
importance_df=DataFrame(importances)
importance_df.index=feat_labels
importance_df.columns=['特征重要性']
importance_sort=importance_df.sort_values(by='特征重要性',ascending=False)
data_train_selected=forest.transform(data_train_clean,threshold=0.001)
data_train_selected_df=DataFrame(data_train_selected)
data_train_selected_df.to_csv('data_train_selected_df',index=False)
#indices=np.argsort(importances)[::-1]
#for f in range(data_train_clean.shape[1]):
#   print("%2d) %-*s %f" % (f+1,30,feat_labels[f],importances[indices[f]]))


"""训练"""
dataselected=pd.read_csv('data_train_selected_df')
dataselected_np=np.array(dataselected)
label_train=pd.read_csv('label_clean.csv')
xgmat=xgb.DMatrix(dataselected_np,label=label_train)
param={}
param['objective'] = 'binary:logistic'
param['eta'] = 0.15
param['max_depth'] = 6
param['eval_metric'] = 'auc'
param['silent'] = 1
param['nthread'] = 16
param['alpha']=50
plst = list(param.items())  
watchlist = [(xgmat,'train')]
# boost 120 trees
num_round = 120
print ('loading data end, start to boost trees')
bst = xgb.train( plst, xgmat, num_round, watchlist )
bst.save_model('disney.model')
#测试
data_test=pd.read_csv('/data/第1题：算法题数据/数据集1_用户标签_本地_测试集.csv')
data_test['是否有出境行为']=np.where(data_test['是否有出境行为']=='是',1,0)
data_test['是否有跨省行为']=np.where(data_test['是否有跨省行为']=='是',1,0)
data_test=data_test.drop(['手机品牌', '手机终端型号', '漫入省份', '漫出省份'],axis=1)
data_test_1=np.array(data_test)
dtest=data_test_1[:,1:340]
xgmat_test = xgb.DMatrix(dtest)
modelfile='disney.model'
bst = xgb.Booster({'nthread':16}, model_file = modelfile)
label_pred=bst.predict(xgmat_test)
a=data_test.iloc[:,0]
label_pred1=DataFrame(label_pred)
df=pd.concat([a,label_pred1],axis=1)
df.to_csv('result.csv',index=False,header=['IMEI','SCORE'],float_format='%.5f')



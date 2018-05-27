# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 18:48:20 2017

@author: loveya
"""
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pandas import DataFrame
from sklearn.cross_validation import train_test_split

data_train=pd.read_csv('/data/第1题：算法题数据/数据集1_用户标签_本地_训练集.csv')
label_train=pd.read_csv('/data/第1题：算法题数据/数据集2_用户是否去过迪士尼_训练集.csv')
data_train['是否有出境行为']=np.where(data_train['是否有出境行为']=='是',1,0)
data_train['是否有跨省行为']=np.where(data_train['是否有跨省行为']=='是',1,0)
data_train=data_train.drop(['手机品牌', '手机终端型号', '漫入省份', '漫出省份'],axis=1)
label_train=label_train['是否去过迪士尼'] 
data=pd.concat([data_train,label_train],axis=1)
data_cleaned=data.dropna()
data_train_clean=data_cleaned.iloc[:,1:339].values
label_train_clean=data_cleaned.iloc[:,339].values
"""划分训练集和测试集"""
X_train,X_test,y_train,y_test=train_test_split(data_train_clean,label_train_clean,test_size=0.25,random_state=0)
X_train_df=DataFrame(X_train)
y_train_df=DataFrame(y_train)
X_test_df=DataFrame(X_test)
y_test_df=DataFrame(y_test)
"""写出文件"""
X_train_df.to_csv('X_train.csv',index=False)
X_test_df.to_csv('X_test.csv',index=False)
y_train_df.to_csv('y_train.csv',index=False)
y_test_df.to_csv('y_test.csv',index=False)
"""读取文件"""
X_train=pd.read_csv('X_train.csv')
X_test=pd.read_csv('X_test.csv')
y_train=pd.read_csv('y_train.csv')
y_test=pd.read_csv('y_test.csv')
X_train=X_train.iloc[:,0:337].values
y_train=y_train.iloc[:,0].values
"""随机森林特征选择"""
feat_labels=data_train.columns[1:338]
forest=RandomForestClassifier(n_estimators=300,random_state=0,n_jobs=-1)
forest.fit(X_train,y_train)
importances=forest.feature_importances_
importance_df=DataFrame(importances)
importance_df.index=feat_labels
importance_df.columns=['importance']
importance_selected=importance_df.ix[importance_df.importance>0.001,0]

X_selected=forest.transform(X_train,threshold=0.001)
X_selected_df=DataFrame(X_selected)
X_selected_df.columns=importance_selected.index
X_selected_df.to_csv('X_selected.csv',index=False)

"""训练"""

X_selected=pd.read_csv('X_selected.csv')
X_selected_np=np.array(X_selected)
y_train=pd.read_csv('y_train.csv')
y_train=y_train.iloc[:,0].values

sum_pos=sum(y_train==1)
sum_neg=sum(y_train==0)

xgmat=xgb.DMatrix(X_selected_np,label=y_train)
param={}
param['objective'] = 'binary:logistic'
param['scale_pos_weight'] = sum_neg/sum_pos
param['eta'] = 0.15
param['max_depth'] = 6
param['eval_metric'] = 'auc'
param['silent'] = 1
param['nthread'] = 16
#param['alpha']=50
plst = list(param.items())  
watchlist = [(xgmat,'train')]
# boost 120 trees
num_round = 120
print ('loading data end, start to boost trees')
bst = xgb.train( plst, xgmat, num_round, watchlist )
bst.save_model('disney.model')

"""测试"""
data_train=pd.read_csv('/data/第1题：算法题数据/数据集1_用户标签_本地_训练集.csv')
data_train=data_train.drop(['手机品牌', '手机终端型号', '漫入省份', '漫出省份'],axis=1)
X_test=pd.read_csv('X_test.csv')
X_test.columns=data_train.columns[1:339]

X_test_selected=X_test[X_selected.columns]
y_test=pd.read_csv('y_test.csv')
y_test_np=np.array(y_test)
X_test_selected_np=np.array(X_test_selected)

xgmat_test = xgb.DMatrix(X_test_selected_np)
modelfile='disney.model'
bst = xgb.Booster({'nthread':16}, model_file = modelfile)
label_pred=bst.predict(xgmat_test)
from sklearn import metrics
test_auc = metrics.roc_auc_score(y_test,label_pred)
"""验证"""
data_test=pd.read_csv('/data/第1题：算法题数据/数据集1_用户标签_本地_测试集.csv')
data_test['是否有出境行为']=np.where(data_test['是否有出境行为']=='是',1,0)
data_test['是否有跨省行为']=np.where(data_test['是否有跨省行为']=='是',1,0)
data_test=data_test.drop(['手机品牌', '手机终端型号', '漫入省份', '漫出省份'],axis=1)


data_test_selected=data_test[X_selected.columns]
data_test_selected_np=np.array(data_test_selected)
xgmat_test = xgb.DMatrix(data_test_selected)
modelfile='disney.model'
bst = xgb.Booster({'nthread':16}, model_file = modelfile)
label_pred=bst.predict(xgmat_test)
a=data_test.iloc[:,0]
label_pred1=DataFrame(label_pred)
df=pd.concat([a,label_pred1],axis=1)
df.to_csv('result.csv',index=False,header=['IMEI','SCORE'],float_format='%.5f')






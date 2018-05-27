# -*- coding: utf-8 -*-
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
import os
import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import gc
import numpy as np
from sklearn.model_selection import GridSearchCV
do_gridsearch = False

# 读取数据
print("Loading the data...")
#x_train = pd.read_csv("数据集1_用户标签_本地_训练集.csv")
x_train = pd.read_csv("processing_xtrain_twofeature.csv")
y_train = pd.read_csv("数据集2_用户是否去过迪士尼_训练集.csv")
x_test_1 = pd.read_csv("数据集1_用户标签_本地_测试集.csv")
x_test = pd.read_csv("processing_xtest_twofeature.csv")

print("Loading finished")

test_index = x_test_1['用户标识'].values
# 正负样本数
pos_sum = (y_train['是否去过迪士尼'] == 1).sum()
neg_sum = y_train.shape[0] - pos_sum

# 去掉漫出漫入省份
#x_train.drop([ '漫入省份'], axis=1, inplace=True)
#x_test.drop(['漫入省份'], axis=1, inplace=True)
# 去除object特征

x_train.fillna(0, inplace=True)
x_test.fillna(0, inplace=True)
# obj_features = x_train.dtypes[x_train.dtypes == 'object'].index
# x_train.drop(obj_features, axis=1, inplace=True)
# x_test.drop(obj_features, axis=1, inplace=True)

x_all = pd.concat([x_train, x_test])

print("Starting LabelEncode")
for c in ['手机终端型号', '手机品牌','漫出省份', '是否有跨省行为','是否有出境行为']:
    lbl = LabelEncoder()
    lbl.fit(list(x_all[c].values))
    x_train[c] = lbl.transform(list(x_train[c].values))
    x_test[c] = lbl.transform(list(x_test[c].values))
del x_all; gc.collect()
print("LabelEncode Finished")



print("Preparing data for training")

#x_train.drop("用户标识", axis=1, inplace=True)
f_name = x_train.columns
x_train = x_train.values


#test_index = x_test['用户标识'].values
#x_test.drop("用户标识", axis=1, inplace=True)
x_test = x_test.values


test = x_test.astype(np.float32, copy=False)
train = x_train.astype(np.float32, copy=False)
labels = np.array(y_train['是否去过迪士尼'])


#d_train = lgb.Dataset(train, labels)


print("Starting splitting")
### we need a test set that we didn't train on to find the best weights for combining the classifiers
'''
sss = StratifiedShuffleSplit(1, test_size=0.05, random_state=1234)
for train_index, test_index in sss.split(train, labels):
    train_x, train_y = train[train_index], labels[train_index]
    test_x, test_y = train[test_index], labels[test_index]
#d_train = lgb.Dataset(train_x, train_y)
'''
train_x = x_train
train_y = labels
test_x = x_test

del train, labels, x_test;gc.collect()


clfs = []



print("Starting lgb")

params = {}
params['max_depth'] = -1
params['num_leaves'] = 270
params['min_data_in_leaf'] = 800
#params['max_bin'] = 10
#params['max_depth'] = 5
params['boosting_type'] = 'gbdt'
params['application'] = 'binary'
params['metric'] = 'auc'
#params['learning_rate'] = 0.1
#params['sub_feature'] = 0.95    # feature_fraction (small values => use very different submodels)
params['sub_feature'] = 0.7
params['bagging_fraction'] = 0.85 # sub_row
#params['bagging_freq'] = 40
#params['bagging_seed'] = 3
params['verbose'] = 0
#params['lambda_l1'] = 10
params['lambda_l1'] = 0.1
#params['lambda_l2'] = 0.1
params['lambda_l2'] = 1
params['is_unbalance'] = True
#params['min_sum_hessian_in_leaf'] = 8
params['learning_rate'] = 0.02
#params['scale_pos_weight'] = np.float(neg_sum)/pos_sum
#print (np.float(neg_sum)/pos_sum)

#params['boosting_type'] = 'gbdt'
#params['application'] = 'binary'
#params['metric'] = 'auc'
#params['verbose'] = 0
#params['learning_rate'] = 0.03
#np.random.seed(0)

#model = lgb.cv(params=params, train_set=d_train, nfold=5, early_stopping_rounds=50, num_boost_round=800, verbose_eval=10)
d_train = lgb.Dataset(train_x, train_y)
#iteration = len(model)
lgb_model = lgb.train(params=params, train_set=d_train, num_boost_round=520, verbose_eval=True)
lgb_pred = lgb_model.predict(test_x)
#print('lgb AUC {score}'.format(score=roc_auc_score(test_y, lgb_model.predict(test_x))))
clfs.append(lgb_model)
#output = pd.DataFrame({'IMEI':test_index, 'SCORE':lgb_pred})
#output.to_csv('result16.csv', index=False, float_format="%.5f")
#print(done!!!)

### usually you'd use xgboost and neural nets here


dtrain = xgb.DMatrix(train_x, train_y)
dtest = xgb.DMatrix(test_x)
xgb_params = {
    'reg_alpha': 1e-2,
    'max_depth': 8,
    'min_child_weight': 3,
    'gamma':0,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'silent': 1,
    'subsample':0.8,
    'colsample_bytree': 0.7,
    'nthread': 4,
    'learning_rate': 0.1,
    'scale_pos_weight': np.float(neg_sum)/pos_sum
}
xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=162)
#print('XGBoost AUC {score}'.format(score=roc_auc_score(test_y, xgb_model.predict(dtest))))

xgb_pred = xgb_model.predict(dtest)
pred = 0.9 * lgb_pred + 0.1 * xgb_pred
output = pd.DataFrame({'IMEI':test_index, 'SCORE':pred})
output.to_csv('result16.csv', index=False, float_format="%.5f")
print("done!!")

clfs.append(xgb_model)


### finding the optimum weights

predictions = []
i = 0
for clf in clfs:
    if i == 1:
        dtest = xgb.DMatrix(test_x)
        predictions.append(clf.predict(dtest))
    else:
        predictions.append(clf.predict(test_x))
    i += 1

def log_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
        final_prediction += weight * prediction

    return roc_auc_score(test_y, final_prediction)

for i in [np.float(k)/10 for k in range(0, 11)]:
# the algorithms need a starting value, right not we chose 0.5 for all weights
# its better to choose many random starting points and run minimize a few timesstarting_values = [0.5] * len(predictions)
    starting_values = [i, 1-i]
    cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
# our weights are bound between 0 and 1
    bounds = [(0, 1)] * len(predictions)

    res =  minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)

    print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
    print('Best Weights: {weights}'.format(weights=res['x']))

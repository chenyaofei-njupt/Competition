import lightgbm as lgb
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import gc
import numpy as np
from sklearn.model_selection import GridSearchCV

do_gridsearch = False

# ========================== 读取数据========================
print("Loading the data...")
x_train = pd.read_csv("数据集1_用户标签_本地_训练集.csv")
y_train = pd.read_csv("数据集2_用户是否去过迪士尼_训练集.csv")
x_test = pd.read_csv("数据集1_用户标签_本地_测试集.csv")
print("Loading finished")

# ========================== 数据预处理 ====================
# 正负样本数
pos_sum = (y_train['是否去过迪士尼'] == 1).sum()
neg_sum = y_train.shape[0] - pos_sum

# 去掉漫出漫入省份
x_train.drop(['漫入省份'], axis=1, inplace=True)
x_test.drop(['漫入省份'], axis=1, inplace=True)

# 填补缺失值
x_train.fillna(-999, inplace=True)
x_test.fillna(-999, inplace=True)


# ========================== 特征工程==========================
# 固定联络圈规模
def feature_eng(x_train):
    # x_train['有无固定联络圈'] = (x_train['固定联络圈规模'] > 0) * 1
    # x_train['固定联络圈规模'] = np.log1p(x_train['固定联络圈规模'])

    # 微信
    # x_train['有无微信'] = (x_train['微信'] > 0) * 1
    # x_train['微信'] = np.log1p(x_train['微信'])

    # 每月的大致刷卡消费次数
    # x_train['有无刷卡消费'] = (x_train['每月的大致刷卡消费次数'] > 0) * 1

    # qq和微信
    x_train['QQ_wechat'] = x_train['QQ'] + x_train['微信']
    # 各类消费次数
    x_train['各类消费的次数'] = x_train['每月的大致刷卡消费次数'] + x_train['手机淘宝'] + x_train['访问购物网站的次数'] + x_train['支付宝钱包']
    # 浏览器
    x_train['浏览器'] = x_train['Safari'] + x_train['IE浏览器'] + x_train['Nexus Browser']
    # 地图
    x_train['地图'] = x_train['高德地图'] + x_train['百度地图']
    # 新闻
    x_train['新闻'] = x_train['腾讯新闻'] + x_train['访问新闻网站的次数']
    # 音乐
    x_train['音乐'] = x_train['访问音乐网站的次数'] + x_train['QQ音乐']
    # 年龄段为5
    x_train['年龄段5'] = (x_train['年龄段'] == 5) * 1
    # 手机品牌和型号
    # x_tain['手机品牌型号']=x_train['手机品牌']+x_train['手机终端型号']
    # 旅游网站和app
    x_train['旅游'] = x_train['携程旅行'] + x_train['访问旅游网站的次数'] + x_train['去哪儿旅行'] + x_train['同程旅游']

    # 大致消费水平
    x_train['大致消费水平_0_2'] = (x_train['大致消费水平'].isin([0, 2])) * 1
    # 影音型网站
    x_train['影音型'] = x_train['访问视频网站的次数'] + x_train['访问音乐网站的次数'] + x_train['访问动漫网站的次数']
    # 股票金融房产网站
    x_train['股票金融房产'] = x_train['访问金融网站的次数'] + x_train['访问股票网站的次数'] + x_train['访问房产网站的次数']
    # 交友型网站
    x_train['交友型网站'] = x_train['访问聊天网站的次数'] + x_train['访问社交网站的次数'] + x_train['访问交友网站的次数'] + x_train['访问通话网站的次数'] + \
                       x_train[
                           '访问论坛网站的次数']
    # 教育知识型网站
    x_train['教育知识型网站'] = x_train['访问问答网站的次数'] + x_train['访问阅读网站的次数'] + x_train['访问新闻网站的次数'] + x_train['访问教育网站的次数']
    # 男生偏好型网站
    x_train['男生偏好型网站'] = x_train['访问体育网站的次数'] + x_train['访问汽车网站的次数'] + x_train['访问游戏网站的次数']
    # 女生偏好型网站
    x_train['女生偏好型网站'] = x_train['访问孕期网站的次数'] + x_train['访问育儿网站的次数'] + x_train['访问健康网站的次数']
    # 出行旅游教育房产汽车
    x_train['出行旅游教育房产汽车'] = x_train['访问旅游网站的次数'] + x_train['访问教育网站的次数'] + x_train['访问房产网站的次数'] + x_train['访问汽车网站的次数']

    # 旅行时必备的网站
    x_train['旅行时必备网站'] = x_train['地图'] + x_train['旅游']

    # 有车一族的有钱人
    x_train['有车一族'] = x_train['车轮查违章'] + x_train['全国违章查询'] + x_train['易到用车'] + x_train['高德导航'] + x_train['汽车之家']

    # 出行交通软件
    x_train['出行交通软件'] = x_train['嘀嘀打车'] + x_train['高铁管家'] + x_train['航旅纵横'] + x_train['快的打车'] + x_train['汽车之家']

    # 国外用户
    x_train['国外用户'] = x_train['GoogleMap'] + x_train['谷歌地图'] + x_train['Facebook'] + x_train['Gmail']

    # 苹果的软件
    x_train['苹果的软件'] = x_train['Safari'] + x_train['苹果iphone股票'] + x_train['iTunes Store'] + x_train['appstore']

    # 英语类软件
    x_train['英语类软件'] = x_train['金山词霸'] + x_train['有道词典'] + x_train['bbc sport'] + x_train['ai jazeera english magazine']
    # 炒股软件
    x_train['炒股软件'] = x_train['大智慧免费炒股软件'] + x_train['同花顺'] + x_train['智远一户通']
    # 最有可能去过的群体
    x_train['最可能群体'] = ((x_train['年龄段'].isin([4, 5])) & (x_train['大致消费水平'] > 5) & (x_train['手机品牌'] == '苹果')) * 1
    # x_train['手机终端型号_苹果'] = (x_train['手机终端型号'].isin(x_train['手机终端型号'].value_counts().index[:7]))*1
    # 无脑的相加
    # x_train['无脑相加'] = x_train['访问IT网站的次数'] + x_train['每月的大致刷卡消费次数'] + x_train['各类消费的次数']
    # 生活类网站
    # x_train['生活类网站'] = x_train['访问健康网站的次数'] + x_train['访问生活网站的次数'] + x_train['访问餐饮网站的次数']
    return x_train


# ===========================定性特征与定量特征的组合===============================
'''
def get_stats_target(df, group_column, target_column, drop_raw_col=False):
    df_old = df[[target_column[0], group_column[0]]].copy()
    grouped = df_old.groupby(group_column)
    the_stats = grouped[target_column].agg(['mean', 'median', 'max', 'min', 'std']).reset_index()

    the_stats.columns = [group_column[0],
                                        '_%s_mean_by_%s' % (target_column[0], group_column[0]),
                                        '_%s_median_by_%s' % (target_column[0], group_column[0]),
                                                '_%s_max_by_%s' % (target_column[0], group_column[0]),
                                                '_%s_min_by_%s' % (target_column[0], group_column[0]),
                                                '_%s_std_by_%s' % (target_column[0], group_column[0])]
    return the_stats.astype(np.int)

stats1  = get_stats_target(x_train, ['年龄段'],  ['固定联络圈规模'])
stats_1  = get_stats_target(x_test, ['年龄段'],  ['固定联络圈规模'])



stats2 = get_stats_target(x_train, ['年龄段'], ['访问IT网站的次数'])
stats_2 = get_stats_target(x_test, ['年龄段'], ['访问IT网站的次数'])

stats3 = get_stats_target(x_train, ['年龄段'], ['QQ'])
stats_3 = get_stats_target(x_test, ['年龄段'], ['QQ'])


stats4 = get_stats_target(x_train, ['年龄段'], ['每月的大致刷卡消费次数'])
stats_4 = get_stats_target(x_test, ['年龄段'], ['每月的大致刷卡消费次数'])

stats5 = get_stats_target(x_train, ['大致消费水平'], ['固定联络圈规模'])
stats_5 = get_stats_target(x_test, ['大致消费水平'], ['固定联络圈规模'])

stats6 = get_stats_target(x_train, ['大致消费水平'], ['访问IT网站的次数'])
stats_6 = get_stats_target(x_test, ['大致消费水平'], ['访问IT网站的次数'])

stats8 = get_stats_target(x_train, ['大致消费水平'], ['每月的大致刷卡消费次数'])
stats_8 = get_stats_target(x_test, ['大致消费水平'], ['每月的大致刷卡消费次数'])

stats9 = get_stats_target(x_train, ['大致消费水平'], ['QQ'])
stats_9 = get_stats_target(x_test, ['大致消费水平'], ['QQ'])

import functools
dfs = [stats1, stats2, stats3, stats4]
dfs2 = [stats5, stats6, stats8, stats9]
dfs_ = [stats_1, stats_2, stats_3, stats_4]
dfs2_ = [stats_5, stats_6, stats_8, stats_9]

merge_train = functools.reduce(lambda left,right:pd.merge(left, right, on='年龄段'), dfs)
merge_train2 = functools.reduce(lambda left,right:pd.merge(left, right, on='大致消费水平'), dfs2)
merge_test = functools.reduce(lambda left,right:pd.merge(left, right, on='年龄段'), dfs_)
merge_test2 = functools.reduce(lambda left,right:pd.merge(left, right, on='大致消费水平'), dfs2_)
x_train = x_train.merge(merge_train, on='年龄段', how='left')
x_train = x_train.merge(merge_train2, on='大致消费水平', how='left')
x_test = x_test.merge(merge_test, on='年龄段', how='left')

#=================执行特征工程函数与编码=================================
print("Starting feature engineering")
x_train = feature_eng(x_train)
x_test = feature_eng(x_test)
print("Starting LabelEncode")
#x_all = pd.concat([x_train, x_test])

for c in ['手机品牌','手机终端型号', '漫出省份','是否有跨省行为','是否有出境行为']:
    lbl = LabelEncoder()
    x_concat = pd.concat([x_train[c], x_test[c]])
    lbl.fit(list(x_concat.values))
    #lbl.fit(list(x_all[c].values))
    x_train[c] = lbl.transform(list(x_train[c].values))
    x_test[c] = lbl.transform(list(x_test[c].values))
print("LabelEncode Finished")


#============================上采样======================
'''
del x_test; gc.collect()
import  pandas_ml as pdml
df = pdml.ModelFrame(x_train.values, target=np.array(y_train['是否去过迪士尼']), columns=x_train.columns)
sampler = df.imbalance.under_sampling.ClusterCentroids()
sampled = df.fit_sample(sampler)
print (sampled.target.value_counts())
'''

#==================数据格式转换===========================
print("Preparing data for training")
x_train.drop("用户标识", axis=1, inplace=True)
f_name = x_train.columns
x_train_df = x_train
x_train = x_train.values
y_train = np.array(y_train['是否去过迪士尼'])


test_index = x_test['用户标识'].values
x_test.drop("用户标识", axis=1, inplace=True)
x_test_df = x_test
x_test = x_test.values



x_test = x_test.astype(np.float32, copy=False)
x_train = x_train.astype(np.float32, copy=False)

d_train = lgb.Dataset(x_train, y_train)

#===================gridsearch========================
if do_gridsearch:
    from sklearn.model_selection import GridSearchCV
    grid_params={
        #'num_leaves': range(20, 80, 10),
        #'min_data_in_leaf': range(200, 900, 200)
    }
    np.random.seed(0)
    model = lgb.LGBMClassifier(boosting_type='gbdt',num_threads=4,  learning_rate=0.1, num_leaves = 45,  objective='binary', scale_pos_weight=np.float(neg_sum)/pos_sum, n_estimators=200)
    grid_model = GridSearchCV(model, grid_params, cv=5, verbose=3,  scoring='roc_auc')
    grid_model.fit(x_train, y_train)
    print("the best_params_ is:", grid_model.best_params_)
    print("the best_score_ is:", grid_model.best_score_)

del x_train; gc.collect()
("Preparing finished")



#================== RUN LIGHTGBT=======================
params = {}
#params['max_depth'] = 4
params['num_leaves'] = 40
params['min_tata_in_leaf'] = 700
import lightgbm as lgb
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import gc
import numpy as np
from sklearn.model_selection import GridSearchCV
do_gridsearch = False

#========================== 读取数据========================
print("Loading the data...")
x_train = pd.read_csv("数据集1_用户标签_本地_训练集.csv")
y_train = pd.read_csv("数据集2_用户是否去过迪士尼_训练集.csv")
x_test = pd.read_csv("数据集1_用户标签_本地_测试集.csv")
print("Loading finished")

#========================== 数据预处理 ====================
# 正负样本数
pos_sum = (y_train['是否去过迪士尼'] == 1).sum()
neg_sum = y_train.shape[0] - pos_sum

# 去掉漫出漫入省份
x_train.drop([ '漫入省份'], axis=1, inplace=True)
x_test.drop(['漫入省份'], axis=1, inplace=True)

#填补缺失值
x_train.fillna(-999, inplace=True)
x_test.fillna(-999, inplace=True)


#========================== 特征工程==========================
# 固定联络圈规模
def feature_eng(x_train):
   # x_train['有无固定联络圈'] = (x_train['固定联络圈规模'] > 0) * 1

 # 微信
# x_train['有无微信'] = (x_train['微信'] > 0) * 1
 # x_train['微信'] = np.log1p(x_train['微信'])

# 每月的大致刷卡消费次数
# x_train['有无刷卡消费'] = (x_train['每月的大致刷卡消费次数'] > 0) * 1

 # qq和微信
 x_train['QQ_wechat'] = x_train['QQ'] + x_train['微信']
 # 各类消费次数
 x_train['各类消费的次数'] = x_train['每月的大致刷卡消费次数'] + x_train['手机淘宝'] + x_train['访问购物网站的次数'] + x_train['支付宝钱包']
 # 浏览器
 x_train['浏览器'] = x_train['Safari'] + x_train['IE浏览器'] + x_train['Nexus Browser']
 # 地图
 x_train['地图'] = x_train['高德地图'] + x_train['百度地图']
 # 新闻
 x_train['新闻'] = x_train['腾讯新闻'] + x_train['访问新闻网站的次数']
 # 音乐
 x_train['音乐'] = x_train['访问音乐网站的次数'] + x_train['QQ音乐']
 # 年龄段为5
 x_train['年龄段5'] = (x_train['年龄段'] == 5) * 1
 # 手机品牌和型号
 # x_tain['手机品牌型号']=x_train['手机品牌']+x_train['手机终端型号']
 # 旅游网站和app
 x_train['旅游'] = x_train['携程旅行'] + x_train['访问旅游网站的次数'] + x_train['去哪儿旅行'] + x_train['同程旅游']

 # 大致消费水平
 x_train['大致消费水平_0_2'] = (x_train['大致消费水平'].isin([0, 2])) * 1
 # 影音型网站
 x_train['影音型'] = x_train['访问视频网站的次数'] + x_train['访问音乐网站的次数'] + x_train['访问动漫网站的次数']
 # 股票金融房产网站
 x_train['股票金融房产'] = x_train['访问金融网站的次数'] + x_train['访问股票网站的次数'] + x_train['访问房产网站的次数']
 # 交友型网站
 x_train['交友型网站'] = x_train['访问聊天网站的次数'] + x_train['访问社交网站的次数'] + x_train['访问交友网站的次数'] + x_train['访问通话网站的次数'] + x_train[
 '访问论坛网站的次数']
 # 教育知识型网站
 x_train['教育知识型网站'] = x_train['访问问答网站的次数'] + x_train['访问阅读网站的次数'] + x_train['访问新闻网站的次数'] + x_train['访问教育网站的次数']
 # 男生偏好型网站
 x_train['男生偏好型网站'] = x_train['访问体育网站的次数'] + x_train['访问汽车网站的次数'] + x_train['访问游戏网站的次数']
 # 女生偏好型网站
 x_train['女生偏好型网站'] = x_train['访问孕期网站的次数'] + x_train['访问育儿网站的次数'] + x_train['访问健康网站的次数']
 # 出行旅游教育房产汽车
 x_train['出行旅游教育房产汽车'] = x_train['访问旅游网站的次数'] + x_train['访问教育网站的次数'] + x_train['访问房产网站的次数'] + x_train['访问汽车网站的次数']

 # 旅行时必备的网站
 x_train['旅行时必备网站'] = x_train['地图'] + x_train['旅游']

 # 有车一族的有钱人
 x_train['有车一族'] = x_train['车轮查违章'] + x_train['全国违章查询'] + x_train['易到用车'] + x_train['高德导航'] + x_train['汽车之家']
                                                                                                                                                           325,13        50%
Last login: Fri Aug 18 17:19:54 on ttys001
xiaoqiangdeMacBook-Pro:~ adslwang4601$ ssh d84227ab1bd76ba4f3f8ad94cacf31@123.59.94.12 -i /Users/adslwang4601/Desktop/liantong/private_key.key -o StrictHostKeyChecking=no
Last login: Fri Aug 18 17:19:34 2017 from 119.129.129.72
[d84227ab1bd76ba4f3f8ad94cacf31@10-9-122-214 ~]$ ssh d84227ab1bd76ba4f3f8ad94cacf31@10.9.131.158
Last login: Fri Aug 18 17:35:46 2017 from 10.9.122.214
[d84227ab1bd76ba4f3f8ad94cacf31@10-9-131-158 ~]$ cd /data
[d84227ab1bd76ba4f3f8ad94cacf31@10-9-131-158 data]$ cd 第1题：算法题数据/
[d84227ab1bd76ba4f3f8ad94cacf31@10-9-131-158 第1题：算法题数据]$ ls
best_032.py
best_045.py
best_434.py
best_result
core.4182
ensembel.py
ensemble_choosing.py
ensemble_weight2.py
ensemble_weight.py
lib.txt
LightGBM.py
matplotlib-d84227ab1bd76ba4f3f8ad94cacf31-9w4ekicj
matplotlib-d84227ab1bd76ba4f3f8ad94cacf31-ozzu0kr9
matplotlib-d84227ab1bd76ba4f3f8ad94cacf31-p8g92d67
matplotlib-d84227ab1bd76ba4f3f8ad94cacf31-qzqmfd2o
matplotlib-d84227ab1bd76ba4f3f8ad94cacf31-ti71cj_v
processing_xtest.csv
processing_xtest_twofeature.csv
processing_xtrain.csv
processing_xtrain.csv_twofeature
processing_xtrain_twofeature.csv
result10.csv
result11.csv
result12.csv
result14.csv
result15.csv
result16.csv
result17.csv
result18.csv
result.csv
sub.csv
svm-output.libsvm
test2
test2.py
test.py
train_df.csv
xgboost_test2.py
数据集1_用户标签_本地_测试集.csv
数据集1_用户标签_本地_训练集.csv
数据集2_用户是否去过迪士尼_训练集.csv
[d84227ab1bd76ba4f3f8ad94cacf31@10-9-131-158 第1题：算法题数据]$ vim best_434.py





[d84227ab1bd76ba4f3f8ad94cacf31@10-9-131-158 第1题：算法题数据]$ ls
best_032.py           ensemble_weight.py                                  processing_xtest.csv              result14.csv       test2
best_045.py           lib.txt                                             processing_xtest_twofeature.csv   result15.csv       test2.py
best_434.py           LightGBM.py                                         processing_xtrain.csv             result16.csv       test.py
best_result           matplotlib-d84227ab1bd76ba4f3f8ad94cacf31-9w4ekicj  processing_xtrain.csv_twofeature  result17.csv       train_df.csv
core.4182             matplotlib-d84227ab1bd76ba4f3f8ad94cacf31-ozzu0kr9  processing_xtrain_twofeature.csv  result18.csv       xgboost_test2.py
ensembel.py           matplotlib-d84227ab1bd76ba4f3f8ad94cacf31-p8g92d67  result10.csv                      result.csv         数据集1_用户标签_本地_测试集.csv
ensemble_choosing.py  matplotlib-d84227ab1bd76ba4f3f8ad94cacf31-qzqmfd2o  result11.csv                      sub.csv            数据集1_用户标签_本地_训练集.csv
ensemble_weight2.py   matplotlib-d84227ab1bd76ba4f3f8ad94cacf31-ti71cj_v  result12.csv                      svm-output.libsvm  数据集2_用户是否去过迪士尼_训练集.csv
[d84227ab1bd76ba4f3f8ad94cacf31@10-9-131-158 第1题：算法题数据]$ vim ensemble_weight2.py
[d84227ab1bd76ba4f3f8ad94cacf31@10-9-131-158 第1题：算法题数据]$ vim best_045.py
[d84227ab1bd76ba4f3f8ad94cacf31@10-9-131-158 第1题：算法题数据]$ vim result.py
[d84227ab1bd76ba4f3f8ad94cacf31@10-9-131-158 第1题：算法题数据]$ vim test.py


    # 有车一族的有钱人
    x_train['有车一族'] = x_train['车轮查违章'] + x_train['全国违章查询'] + x_train['易到用车'] + x_train['高德导航'] + x_train['汽车之家']

    # 出行交通软件
    x_train['出行交通软件'] = x_train['嘀嘀打车'] + x_train['高铁管家'] + x_train['航旅纵横'] + x_train['快的打车'] + x_train['汽车之家']

    # 国外用户
    x_train['国外用户'] = x_train['GoogleMap'] + x_train['谷歌地图'] + x_train['Facebook'] + x_train['Gmail']

    # 苹果的软件
    x_train['苹果的软件'] =  x_train['Safari'] + x_train['苹果iphone股票'] + x_train['iTunes Store'] + x_train['appstore']

    # 英语类软件
    x_train['英语类软件'] = x_train['金山词霸'] + x_train['有道词典'] + x_train['bbc sport'] + x_train['ai jazeera english magazine']
    # 炒股软件
    x_train['炒股软件'] = x_train['大智慧免费炒股软件'] + x_train['同花顺'] + x_train['智远一户通']
    # 最有可能去过的群体
    x_train['最可能群体'] = ((x_train['年龄段'].isin([4, 5])) & (x_train['大致消费水平'] > 5) & (x_train['手机品牌'] == '苹果'))*1
   # x_train['手机终端型号_苹果'] = (x_train['手机终端型号'].isin(x_train['手机终端型号'].value_counts().index[:7]))*1
    # 无脑的相加
   # x_train['无脑相加'] = x_train['访问IT网站的次数'] + x_train['每月的大致刷卡消费次数'] + x_train['各类消费的次数']
    # 生活类网站
   # x_train['生活类网站'] = x_train['访问健康网站的次数'] + x_train['访问生活网站的次数'] + x_train['访问餐饮网站的次数']
    return x_train

#===========================定性特征与定量特征的组合===============================
'''


def get_stats_target(df, group_column, target_column, drop_raw_col=False):
    df_old = df[[target_column[0], group_column[0]]].copy()
    grouped = df_old.groupby(group_column)
    the_stats = grouped[target_column].agg(['mean', 'median', 'max', 'min', 'std']).reset_index()

    the_stats.columns = [group_column[0],
                         '_%s_mean_by_%s' % (target_column[0], group_column[0]),
                         '_%s_median_by_%s' % (target_column[0], group_column[0]),
                         '_%s_max_by_%s' % (target_column[0], group_column[0]),
                         '_%s_min_by_%s' % (target_column[0], group_column[0]),
                         '_%s_std_by_%s' % (target_column[0], group_column[0])]
    return the_stats.astype(np.int)


stats1 = get_stats_target(x_train, ['年龄段'], ['固定联络圈规模'])
stats_1 = get_stats_target(x_test, ['年龄段'], ['固定联络圈规模'])

stats2 = get_stats_target(x_train, ['年龄段'], ['访问IT网站的次数'])
stats_2 = get_stats_target(x_test, ['年龄段'], ['访问IT网站的次数'])

stats3 = get_stats_target(x_train, ['年龄段'], ['QQ'])
stats_3 = get_stats_target(x_test, ['年龄段'], ['QQ'])

stats4 = get_stats_target(x_train, ['年龄段'], ['每月的大致刷卡消费次数'])
stats_4 = get_stats_target(x_test, ['年龄段'], ['每月的大致刷卡消费次数'])

stats5 = get_stats_target(x_train, ['大致消费水平'], ['固定联络圈规模'])
stats_5 = get_stats_target(x_test, ['大致消费水平'], ['固定联络圈规模'])

stats6 = get_stats_target(x_train, ['大致消费水平'], ['访问IT网站的次数'])
stats_6 = get_stats_target(x_test, ['大致消费水平'], ['访问IT网站的次数'])

stats8 = get_stats_target(x_train, ['大致消费水平'], ['每月的大致刷卡消费次数'])
stats_8 = get_stats_target(x_test, ['大致消费水平'], ['每月的大致刷卡消费次数'])

stats9 = get_stats_target(x_train, ['大致消费水平'], ['QQ'])
stats_9 = get_stats_target(x_test, ['大致消费水平'], ['QQ'])

import functools
dfs = [stats1, stats2, stats3, stats4]
dfs2 = [stats5, stats6, stats8, stats9]
dfs_ = [stats_1, stats_2, stats_3, stats_4]
dfs2_ = [stats_5, stats_6, stats_8, stats_9]

merge_train = functools.reduce(lambda left,right:pd.merge(left, right, on='年龄段'), dfs)
merge_train2 = functools.reduce(lambda left,right:pd.merge(left, right, on='大致消费水平'), dfs2)
merge_test = functools.reduce(lambda left,right:pd.merge(left, right, on='年龄段'), dfs_)
merge_test2 = functools.reduce(lambda left,right:pd.merge(left, right, on='大致消费水平'), dfs2_)
x_train = x_train.merge(merge_train, on='年龄段', how='left')
x_train = x_train.merge(merge_train2, on='大致消费水平', how='left')
x_test = x_test.merge(merge_test, on='年龄段', how='left')
x_test = x_test.merge(merge_test2, on='大致消费水平', how='left')
'''

#=================执行特征工程函数与编码=================================
print("Starting feature engineering")
x_train = feature_eng(x_train)
x_test = feature_eng(x_test)
print("Starting LabelEncode")
#x_all = pd.concat([x_train, x_test])

for c in ['手机品牌','手机终端型号', '漫出省份','是否有跨省行为','是否有出境行为']:
    lbl = LabelEncoder()
    x_concat = pd.concat([x_train[c], x_test[c]])
    lbl.fit(list(x_concat.values))
    #lbl.fit(list(x_all[c].values))
    x_train[c] = lbl.transform(list(x_train[c].values))
    x_test[c] = lbl.transform(list(x_test[c].values))
print("LabelEncode Finished")

#============================上采样======================
'''
del x_test; gc.collect()
import  pandas_ml as pdml
df = pdml.ModelFrame(x_train.values, target=np.array(y_train['是否去过迪士尼']), columns=x_train.columns)
sampler = df.imbalance.under_sampling.ClusterCentroids()
sampled = df.fit_sample(sampler)
print (sampled.target.value_counts())
'''

#==================数据格式转换===========================
print("Preparing data for training")
x_train.drop("用户标识", axis=1, inplace=True)
f_name = x_train.columns
x_train_df = x_train
x_train = x_train.values
y_train = np.array(y_train['是否去过迪士尼'])


test_index = x_test['用户标识'].values
x_test.drop("用户标识", axis=1, inplace=True)
x_test_df = x_test
x_test = x_test.values



x_test = x_test.astype(np.float32, copy=False)
x_train = x_train.astype(np.float32, copy=False)

d_train = lgb.Dataset(x_train, y_train)

#===================gridsearch========================
if do_gridsearch:
    from sklearn.model_selection import GridSearchCV
    grid_params={
        #'num_leaves': range(20, 80, 10),
        #'min_data_in_leaf': range(200, 900, 200)
    }
    np.random.seed(0)
    model = lgb.LGBMClassifier(boosting_type='gbdt',num_threads=4,  learning_rate=0.1, num_leaves = 45,  objective='binary', scale_pos_weight=np.float(neg_sum)/pos_sum, n_estimators=200)
    grid_model = GridSearchCV(model, grid_params, cv=5, verbose=3,  scoring='roc_auc')
    grid_model.fit(x_train, y_train)
    print("the best_params_ is:", grid_model.best_params_)
    print("the best_score_ is:", grid_model.best_score_)

del x_train; gc.collect()
("Preparing finished")



#================== RUN LIGHTGBT=======================
params = {}
#params['max_depth'] = 4
params['num_leaves'] = 40
params['min_data_in_leaf'] = 700
#params['max_bin'] = 10
#params['max_depth'] = 5
params['boosting_type'] = 'gbdt'
params['application'] = 'binary'
params['metric'] = 'auc'
#params['learning_rate'] = 0.1
params['sub_feature'] = 0.95    # feature_fraction (small values => use very different submodels)
params['bagging_fraction'] = 0.85 # sub_row
#params['bagging_freq'] = 40
#params['bagging_seed'] = 3
params['verbose'] = 0
params['lambda_l1'] = 10
params['lambda_l2'] = 0.1
params['is_unbalance'] = True
#params['min_sum_hessian_in_leaf'] = 8
params['learning_rate'] = 0.05
#params['scale_pos_weight'] = np.float(neg_sum)/pos_sum
#print (np.float(neg_sum)/pos_sum)

#params['boosting_type'] = 'gbdt'
#params['application'] = 'binary'
#params['metric'] = 'auc'
#params['verbose'] = 0
#params['learning_rate'] = 0.03
#np.random.seed(0)
print("Oringinal CV")
#lgb.cv(params=params, train_set=d_train, nfold=5, early_stopping_rounds=50, num_boost_round=800, verbose_eval=10)
#print (len(model))
model = lgb.train(params=params, train_set=d_train, num_boost_round=400, verbose_eval=True)
f_score = model.feature_importance()
f_score_df = pd.DataFrame({'name':f_name, 'score':f_score})
#print(f_score_df.head(50))
f_score_df =  f_score_df.sort_values(by='score',ascending=False)
print (f_score_df.iloc[:60, :])

#=====================特征选择与再训练===============================
print("Starting the feature selection training")
for n in [255]:
    f_300 = f_score_df['name'].iloc[:n].values
    x_train_df[f_300].to_csv('processing_xtrain_twofeature.csv',index=False)
    x_train_300 = x_train_df[f_300].values.astype(np.float32)
    #y_train.to_csv('processing_ytrain.csv', index=False)
    d_train = lgb.Dataset(x_train_300, y_train)

   # model = lgb.cv(params=params, train_set=d_train, early_stopping_rounds=50, num_boost_round=1000, verbose_eval=20)
   # model = lgb.train(params=params, train_set=d_train, num_boost_round=320, verbose_eval=True)

#====================预测与保存==============================
print("Starting predict")
print (x_test_df[f_300].shape[1])
x_test_df[f_300].to_csv('processing_xtest_twofeature.csv', index=False)
print("Saving csv done")
lightgbm_pred = model.predict(x_test_df[f_300].values.astype(np.float32))

output = pd.DataFrame({'IMEI':test_index, 'SCORE':lightgbm_pred})
output.to_csv('result17.csv', index=False, float_format="%.5f")


#params['max_bin'] = 10
#params['max_depth'] = 5
params['boosting_type'] = 'gbdt'
params['application'] = 'binary'
params['metric'] = 'auc'
#params['learning_rate'] = 0.1
params['sub_feature'] = 0.95    # feature_fraction (small values => use very different submodels)
params['bagging_fraction'] = 0.85 # sub_row
#params['bagging_freq'] = 40
#params['bagging_seed'] = 3
params['verbose'] = 0
params['lambda_l1'] = 10
params['lambda_l2'] = 0.1
params['is_unbalance'] = True
#params['min_sum_hessian_in_leaf'] = 8
params['learning_rate'] = 0.05
#params['scale_pos_weight'] = np.float(neg_sum)/pos_sum
#print (np.float(neg_sum)/pos_sum)

#params['boosting_type'] = 'gbdt'
#params['application'] = 'binary'
#params['metric'] = 'auc'
#params['verbose'] = 0
#params['learning_rate'] = 0.03
#np.random.seed(0)
print("Oringinal CV")
#lgb.cv(params=params, train_set=d_train, nfold=5, early_stopping_rounds=50, num_boost_round=800, verbose_eval=10)
#print (len(model))
model = lgb.train(params=params, train_set=d_train, num_boost_round=400, verbose_eval=True)
f_score = model.feature_importance()
f_score_df = pd.DataFrame({'name':f_name, 'score':f_score})
#print(f_score_df.head(50))
f_score_df =  f_score_df.sort_values(by='score',ascending=False)
print (f_score_df.iloc[:60, :])

#=====================特征选择与再训练===============================
print("Starting the feature selection training")
for n in [255]:
    f_300 = f_score_df['name'].iloc[:n].values
    x_train_df[f_300].to_csv('processing_xtrain_twofeature.csv',index=False)
    x_train_300 = x_train_df[f_300].values.astype(np.float32)
    #y_train.to_csv('processing_ytrain.csv', index=False)
    d_train = lgb.Dataset(x_train_300, y_train)

   # model = lgb.cv(params=params, train_set=d_train, early_stopping_rounds=50, num_boost_round=1000, verbose_eval=20)
   # model = lgb.train(params=params, train_set=d_train, num_boost_round=320, verbose_eval=True)

#====================预测与保存==============================
print("Starting predict")
print (x_test_df[f_300].shape[1])
x_test_df[f_300].to_csv('processing_xtest_twofeature.csv', index=False)
print("Saving csv done")
lightgbm_pred = model.predict(x_test_df[f_300].values.astype(np.float32))
output = pd.DataFrame({'IMEI':test_index, 'SCORE':lightgbm_pred})
output.to_csv('result17.csv', index=False, float_format="%.5f")
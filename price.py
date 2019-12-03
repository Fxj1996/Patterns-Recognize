import numpy as np  # linear algebra
import pandas as pd  #
from datetime import datetime
from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#显示矩阵对象的维数，核查是否导入成功
# print("Train set size:", train.shape)
# print("Test set size:", test.shape)

#记录运算开始的时间
print('START data processing', datetime.now(), )

#将train矩阵中的'Id'列删除（原地删除，故将inplace设为true），因为原始数据中的数据索引和预测模型的构建没有关系。
#test矩阵类似。
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)

train = train[train.GrLivArea < 4000]
#由于删去了部分行，故此时train矩阵中的index列并不连续。使用reset_index命令，在固定非index数据的顺序的前提下（inplace=True），重新对index编号（drop=True）。
train.reset_index(drop=True, inplace=True)
# log1p就是log(1+x)，用来对房价数据进行数据预处理，它的好处是转化后的数据更加服从高斯分布，有利于后续的分类结果。
train["SalePrice"] = np.log1p(train["SalePrice"])
y = train['SalePrice'].reset_index(drop=True) ##单独取出训练数据中的房价列信息，存入y对象
#沿着水平的方向寻找列名为'SalePrice'的列（们），把它们对应的列统统删掉。得到了单纯的特征矩阵，存入train_features对象中。
train_features = train.drop(['SalePrice'], axis=1)
#test本来就没有房价列，所以它本来就是单纯的特征矩阵。
test_features = test

#将训练数据中的特征矩阵和测试数据中的特征矩阵合并，并对合并后的矩阵index重新编号
features = pd.concat([train_features, test_features]).reset_index(drop=True)
##合并训练数据特征矩阵与测试数据特征矩阵，以便统一进行特征处理-【结束】##
print("剔除训练数据中的极端值后，将其特征矩阵和测试数据中的特征矩阵合并，维度为:",features.shape)

#对于列名为'MSSubClass'、'YrSold'、'MoSold'的特征列，将列中的数据类型转化为string格式。
features['MSSubClass'] = features['MSSubClass'].apply(str)
features['YrSold'] = features['YrSold'].astype(str)
features['MoSold'] = features['MoSold'].astype(str)

#按照以下各个特征列的实际情况，依次处理各个特征列中的空值（.fillna()方法）
features['Functional'] = features['Functional'].fillna('Typ') #空值填充为str型数据'Typ'
features['Electrical'] = features['Electrical'].fillna("SBrkr") #空值填充为str型数据"SBrkr"
features['KitchenQual'] = features['KitchenQual'].fillna("TA") #空值填充为str型数据"TA"
features["PoolQC"] = features["PoolQC"].fillna("None") #空值填充为str型数据"None"

#对于列名为'Exterior1st'、'Exterior2nd'、'SaleType'的特征列，使用列中的众数填充空值。
#	1.先查找数据列中的众数：使用df.mode()[]方法
#	  解释：df.mode(0或1,0表示对列查找，1表示对行查找)[需要查找众数的df列的index（就是df中的第几列）]，将返回数据列中的众数
#	2.使用.fillna()方法进行填充
features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0]) 
features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])

#对于列名为'GarageYrBlt', 'GarageArea', 'GarageCars'的特征列，使用0填充空值。
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    features[col] = features[col].fillna(0)

#对于列名为'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'的特征列，使用字符串'None'填充空值。
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    features[col] = features[col].fillna('None')

#对于列名为'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'的特征列，使用字符串'None'填充空值。
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    features[col] = features[col].fillna('None')

#聚合函数（按某一列关键字分组）groupby，它的特点是：将返回与传入方法的矩阵维度相同的单个序列。
#transform是与groupby（pandas中最有用的操作之一）通常组合使用，它对传入方法的矩阵进行维度不变的变换。具体变换方法写在括号中，通常会使用匿名函数,对传入矩阵的所有元素进行操作。
#对于featurgroupbyes矩阵，按照'MSSubClass'列中的元素分布进行分组，被分组的数据列是'MSZoning'列。feature.groupby(被作为索引的列的名称)[被分组的数据列的名称]
#features.('MSSubClass')['MSZoning']后，得到的是一个以'MSSubClass'为索引，以'MSZoning'为数据列的矩阵。
#.transform()方法将对'MSZoning'数据列进行()内的变换，它将返回和传入矩阵同样维度的矩阵。
#括号内是匿名函数，将对传入矩阵中的空值进行填充，使用的填充元素是传入矩阵中的众数。
features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

#判断出features矩阵中列为对象的列，将列名存入object数祖。对于features矩阵中的各个列对象，将其列中的空值填充为'None'
objects = []
for i in features.columns:
    if features[i].dtype == object:
        objects.append(i)
features.update(features[objects].fillna('None'))

#使用传入矩阵（'LotFrontage'列）的中位数对传入矩阵中的空值进行填充。
#先以'Neighborhood'为标签，以'LotFrontage'为被汇总序列。然后使用被汇总序列中的中位数，对原始矩阵'LotFrontage'列中的空值进行填充。
#transform的特性是同维操作，最后输出结果的顺序和原始数据在序号上完全匹配。
features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

#对于整型和浮点型数据列，使用0填充其中的空值。
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics = []
for i in features.columns:
    if features[i].dtype in numeric_dtypes:
        numerics.append(i)
features.update(features[numerics].fillna(0))
###############################填充空值-【结束】##########################

######################数字型数据列偏度校正-【开始】#######################
#偏度是统计数据分布偏斜方向和程度的度量，是统计数据分布非对称程度的数字特征。亦称偏态、偏态系数。
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics2 = []
for i in features.columns:
    if features[i].dtype in numeric_dtypes:
        numerics2.append(i)
skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

#以0.5作为基准，统计偏度超过此数值的高偏度分布数据列，获取这些数据列的index。
high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

#对高偏度数据进行处理，将其转化为正态分布。
for i in skew_index:
    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))

######################数字型数据列偏度校正-【结束】#######################

######################特征删除和融合创建新特征-【开始】###################
features = features.drop(['Utilities', 'Street', 'PoolQC','LandSlope','RoofMatl','Heating'], axis=1)


#融合多个特征，生成新特征。
features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']

features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +
                                 features['1stFlrSF'] + features['2ndFlrSF'])

features['Total_Bathrooms'] = (features['FullBath'] + (0.2 * features['HalfBath']) +
                               features['BsmtFullBath'] + (0.8 * features['BsmtHalfBath']))

features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +
                              features['EnclosedPorch'] + features['ScreenPorch'] +
                              features['WoodDeckSF'])

#简化特征。对于某些分布单调（比如100个数据中有99个的数值是0.9，另1个是0.1）的数字型数据列，进行01取值处理。
features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
######################特征删除和融合创建新特征-【结束】###################

####################特征投影、特征矩阵拆解和截取-【开始】#################
#使用.get_dummies()方法对特征矩阵进行类似“坐标投影”操作。获得在新空间下的特征表达。
final_features = pd.get_dummies(features).reset_index(drop=True)
print("使用get_dummies()方法“投影”特征矩阵，即分解出更多特征，得到更多列。投影后的特征矩阵维度为:",final_features.shape)

#进行特征空间降阶。截取前len(y)行，存入X阵（因为之前进行了训练数据和测试数据的合并，所以从合并矩阵中取出前len(y)行，就得到了训练数据集的处理后的特征矩阵）。
#截取剩余部分，即从序号为len(y)的行开始，至矩阵结尾的各行，存入X_sub阵。此为完成特征变换后的测试集特征矩阵。
#注：len(df)是行计数方法
X = final_features.iloc[:len(y), :]	#y是列向量，存储了训练数据中的房价列信息。截取后得到的X阵的维度是len(y)*(final_features的列数)。
X_sub = final_features.iloc[len(y):, :]#使用len命令，求矩阵X的长度，得到的是矩阵对象的长度，即有矩阵中有多少列，而不是每列上有多少行。

#在新生特征空间中，剔除X阵和y阵中有着极端值的各行数据（因为X和y阵在水平方向上是一致的，所以要一起删除同样的行）。outliers数值中给出了极端值的列序号。
#df.drop(df.index[序号])将删除指定序号的各行。再使用=对df覆值。
outliers = [30, 88, 462, 631, 1322]
X = X.drop(X.index[outliers])#因为X阵是经过对特征矩阵进行类似“坐标投影”操作后得到的，列向量y中的行号对应着X阵中的列号。
y = y.drop(y.index[outliers])
#############特征投影、特征矩阵拆解和截取-【结束】#################

####################################机器学习-【开始】###########################################
#设置k折交叉验证的参数。
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

#定义均方根误差（Root Mean Squared Error ，RMSE）
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

#创建模型评分函数，根据不同模型的表现打分
def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))
    return rmse

#############个体机器学习模型的创建（即模型声明和参数设置）-【开始】############
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

#定义ridge(L2)岭回归模型（使用二范数作为正则化项。不论是使用一范数还是二范数，正则化项的引入均是为了降低过拟合风险。）
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))

#定义LASSO收缩模型（使用L1范数作为正则化项）（由于对目标函数的求解结果中将得到很多的零分量，它也被称为收缩模型。）
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))

#定义SVM支持向量机模型                                     
svr = make_pipeline(RobustScaler(), SVR(C=30, epsilon= 0.008, gamma=0.0003))

#定义GB梯度提升模型（展开到一阶导数）									
gbr = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)

#定义lightgbm模型									
lightgbm = LGBMRegressor(objective='regression',
                                       num_leaves=4,
                                       learning_rate=0.01,
                                       n_estimators=1000,
                                       max_bin=200,
                                       bagging_fraction=0.75,
                                       bagging_freq=5,
                                       bagging_seed=7,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=7,
                                       verbose=-1                                       #min_data_in_leaf=2,
                                       )

#定义xgboost模型（展开到二阶导数）
xgboost = XGBRegressor(learning_rate=0.01, n_estimators=1000,
                                     max_depth=3, min_child_weight=3,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)
#############个体机器学习模型的创建（即模型声明和参数设置）-【结束】############

###########################集成多个个体学习器-【开始】##########################

stack_gen = StackingCVRegressor(regressors=(ridge, lasso, gbr, xgboost, lightgbm),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)
###########################集成多个个体学习器-【结束】##########################                             

############################进行交叉验证打分-【开始】###########################
#进行交叉验证，并对不同模型的表现打分
#（由于是交叉验证，将使用不同的数据集对同一模型进行评分，故每个模型对应一个得分序列。展示模型得分序列的平均分、标准差）
print('进行交叉验证，计算不同模型的得分TEST score on CV')

score = cv_rmse(ridge)
print("二范数rideg岭回归模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(lasso)
print("一范数LASSO收缩模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(svr)
print("SVR支持向量机模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(lightgbm)
print("lightgbm轻梯度提升模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(gbr)
print("gbr梯度提升回归模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(xgboost)
print("xgboost模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

############################进行交叉验证打分-【结束】###########################

#########使用训练数据特征矩阵作为输入，训练数据对数处理后的预测房价作为输出，进行各个模型的训练-【开始】#########
#开始集合所有模型，使用stacking方法
print(datetime.now(), '对stack_gen集成器模型进行参数训练')
stack_gen_model = stack_gen.fit(np.array(X), np.array(y))

print(datetime.now(), '对一范数lasso收缩模型进行参数训练')
lasso_model_full_data = lasso.fit(X, y)

print(datetime.now(), '对二范数ridge岭回归模型进行参数训练')
ridge_model_full_data = ridge.fit(X, y)

print(datetime.now(), '对svr支持向量机模型进行参数训练')
svr_model_full_data = svr.fit(X, y)

print(datetime.now(), '对GradientBoosting梯度提升模型进行参数训练')
gbr_model_full_data = gbr.fit(X, y)

print(datetime.now(), '对xgboost二阶梯度提升模型进行参数训练')
xgb_model_full_data = xgboost.fit(X, y)

print(datetime.now(), '对lightgbm轻梯度提升模型进行参数训练')
lgb_model_full_data = lightgbm.fit(X, y)
#########使用训练数据特征矩阵作为输入，训练数据对数处理后的预测房价作为输出，进行各个模型的训练-【结束】#########

############################进行交叉验证打分-【结束】###########################

########定义个体学习器的预测值融合函数，检测预测值融合策略的效果-【开始】#######
#综合多个模型产生的预测值，作为多模型组合学习器的预测值
def blend_models_predict(X):
    return ((0.1 * lasso_model_full_data.predict(X)) + \
            (0.1 * ridge_model_full_data.predict(X)) + \
            (0.15 * svr_model_full_data.predict(X)) + \
            (0.15 * gbr_model_full_data.predict(X)) + \
            (0.1 * xgb_model_full_data.predict(X)) + \
            (0.1 * lgb_model_full_data.predict(X)) + \
            (0.3 * stack_gen_model.predict(np.array(X))))

#打印在上述模型配比下，多模型组合学习器的均方根对数误差（Root Mean Squared Logarithmic Error ，RMSLE）
#使用训练数据对创造的模型进行k折交叉验证，以训练创造出的模型的参数配置。交叉验证训练过程结束后，将得到模型的参数配置。使用得出的参数配置下，在全体训练数据上进行验证，验证模型对全体训练数据重构的误差。
print('融合后的训练模型对原数据重构时的均方根对数误差RMSLE score on train data:')
print(rmsle(y, blend_models_predict(X)))
########定义个体学习器的预测值融合函数，检测预测值融合策略的效果-【结束】#######

########将测试集的特征矩阵作为输入，传入训练好的模型，得出的输出写入.csv文件的第2列-【开始】########
print('使用测试集特征进行房价预测 Predict submission', datetime.now(),)
submission = pd.read_csv("sample_submission.csv")
#函数注释：.iloc[:,1]是基于索引位来选取数据集，[索引1:索引2]，左闭右开。
submission.iloc[:,1] = np.floor(np.expm1(blend_models_predict(X_sub))) #返回不大于输入参数的最大整数

q1 = submission['SalePrice'].quantile(0.005)
q2 = submission['SalePrice'].quantile(0.995)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
########将测试集的特征矩阵作为输入，传入训练好的模型，得出的输出写入.csv文件的第2列-【结束】########

#以csv文件的形式输出预测值
submission.to_csv("House_price_submission.csv", index=False)
print('融合结果.csv文件输出成功 Save submission', datetime.now())

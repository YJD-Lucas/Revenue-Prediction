import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime as dt
from scipy.stats import shapiro
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C  # RBF就是高斯核函数
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

train = pd.read_csv("./train2.csv")
# test = pd.read_csv("./test.csv")
# print("Train :",train.shape)
# print("Test:",test.shape)
num_train = train.shape[0]
# num_test = test.shape[0]
# print(num_train, num_test)

# 训练数据与检测数据的组合
# data = pd.concat((train.loc[:, "Open Date" : "P37"],
#                   test.loc[:, "Open Date" : "P37"]), ignore_index=True)
# 缺失数据检测
data = train
print(data.tail())
print(data.columns.values.tolist())
print(data.isnull().sum().T)
missing_df = data.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.loc[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')

# ind = np.arange(missing_df.shape[0])
# width = 0.9
# fig, ax = plt.subplots(figsize=(12,18))
# rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
# ax.set_yticks(ind)
# ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
# ax.set_xlabel("Count of missing values")
# ax.set_title("Number of missing values in each column")
# plt.show()

# train 数据预处理
all_diff = []
datenow = '01/01/2015'
for date in data["Open Date"]:
    diff = dt.strptime(datenow, "%m/%d/%Y") - dt.strptime(date, "%m/%d/%Y")
    all_diff.append(int(diff.days/100))

data['Days_from_open'] = pd.Series(all_diff)

# Drop Open Date Column
data = data.drop('Open Date', 1)
data = data.drop('Id', 1)

data['City Group'] = data['City Group'].map({'Other':0,'Big Cities':1})
data["Type"] = data["Type"].map({"FC":0, "IL":1, "DT":2, "MB":3})

distinct_cities = train.loc[:, "City"].unique()
print(distinct_cities)
print(data.head())

# 相关性分析
# Correlation = data[:].corr()
# Correlation = Correlation.sort_values('revenue',axis=0)
# print(abs(Correlation.revenue))
Correlation = data[['revenue', 'City Group', 'Type', 'Days_from_open', 'P1', 'P2', 'P3', 'P4', 'P5']].corr()
mask = np.array(Correlation)
mask[np.tril_indices_from(mask)] = False
fig1 = plt.figure()
fig1.set_size_inches(10, 10)
ax1 = fig1.add_subplot(1, 1, 1)
sns.heatmap(Correlation, mask=mask, cbar=True, annot=True, ax=ax1)
ax1.set(title="Correlation Analysis")
ax1.set_yticklabels(['revenue', 'City Group', 'Type', 'open Time', 'P1', 'P2', 'P3', 'P4', 'P5'])
ax1.set_xticklabels(['revenue', 'City Group', 'Type', 'open Time', 'P1', 'P2', 'P3', 'P4', 'P5'])

Correlation2 = data[['revenue', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P30']].corr()
mask = np.array(Correlation2)
mask[np.tril_indices_from(mask)] = False
fig2 = plt.figure()
fig2.set_size_inches(10, 10)
ax2 = fig2.add_subplot(1, 1, 1)
sns.heatmap(Correlation2, mask=mask, cbar=True, annot=True, ax=ax2)
ax2.set(title="Correlation Analysis")
ax2.set_yticklabels(['revenue', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P30'])
ax2.set_xticklabels(['revenue', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P30'])
plt.show()

# 验证revenue是否满足高斯分布
plt.rcParams['figure.figsize'] = (16.0, 6.0)
pvalue_before = shapiro(data["revenue"])[1]
pvalue_after = shapiro(np.log(data["revenue"]))[1]
graph_data = pd.DataFrame(
        {
            ("Revenue\n P-value:" + str(pvalue_before)) : data["revenue"],
            ("Log(Revenue)\n P-value:" + str(pvalue_after)) : np.log(data["revenue"])
        }
    )
graph_data.hist()

data["revenue"] = np.log(data["revenue"])

# 先根据相关系数进行特征选择，特征总数为41维，这里假定我们选择特征维数为十维
# （后面再根据联合互信息和加权自采样进行特征维数的选择）
# FeatureNames = ['Days_from_open','City Group', 'P2', 'P28', 'P6', 'P29', 'P13', 'Type', 'P21', 'P11']
FeatureNames = ['Days_from_open','City Group', 'P2', 'P28', 'P6', 'P13', 'Type', 'P21', 'P11']
yLabels = data['revenue']
DataTrain = data[FeatureNames]

# 1.Linear Regression model
# L_Model = LinearRegression()
# L_Model.fit(X=DataTrain, y=yLabels)
# preds = L_Model.predict(X=DataTrain)

# 2.GPR model
# kernel = C(0.1, (0.001, 0.1))*RBF(0.5, (1e-4, 10))
# reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
# reg.fit(X=DataTrain, y=yLabels)
# preds = reg.predict(X=DataTrain)

# 3.RandomForestRegressor model
R_Model = RandomForestRegressor(n_estimators=100)
R_Model.fit(DataTrain,yLabels)
preds = R_Model.predict(X=DataTrain)

# 4.Ensemble Model - Gradient Boost
# G_Model = GradientBoostingRegressor(n_estimators=4000, alpha=0.01)
# G_Model.fit(DataTrain, yLabels)
# preds = G_Model.predict(X=DataTrain)

print(yLabels)
print(preds)

fig3 = plt.figure()
fig3.set_size_inches(10, 10)
ax3 = fig3.add_subplot(1, 1, 1)
x = np.array(range(137))
sns.pointplot(x=x, y=yLabels, label='fact', color='blue', ax=ax3)
sns.pointplot(x=x, y=preds, label='prediction', color='red', ax=ax3)
ax3.set(xlabel='ID', ylabel='revenue',title="Random Forest Regression",label='big')
# ax3.set(xlabel='ID', ylabel='revenue',title="GPR model",label='big')
# ax3.set(xlabel='ID', ylabel='revenue',title="Linear Regression model",label='big')
# ax3.set(xlabel='ID', ylabel='revenue',title="Gradient Boost",label='big')
plt.legend(['fact', 'prediction'])
plt.show()


# 根据要求建立评估模型
def rmsle(y, y_, convertExp=True):
    if convertExp:
        y = np.exp(y),
        y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))


print("RMSLE Value For model: ", rmsle(np.exp(yLabels), np.exp(preds), False))

residual = yLabels-preds


def Hsic(Kx, Ky):
    Kxy = np.dot(Kx, Ky)
    n = Kxy.shape[0]
    h = np.trace(Kxy) / n ** 2 + np.mean(Kx) * np.mean(Ky) - 2 * np.mean(Kxy) / n
    return h * n ** 2 / (n - 1) ** 2


K_yLabels = np.expand_dims(yLabels, 0) - np.expand_dims(yLabels, 1)
K_yLabels = np.exp(- K_yLabels**2)

K_residual = np.expand_dims(residual, 0) - np.expand_dims(residual, 1)
K_residual = np.exp(- K_residual**2)
print(Hsic(K_residual,K_yLabels))
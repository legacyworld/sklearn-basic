# 課題 4.1 リッジ回帰とラッソの比較
# Youtubeでの解説：第5回(1) 12分50秒あたり
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score


#scikit-leanよりワインのデータをインポートする
df= pd.read_csv('winequality-red.csv',sep=';')
# 目標値であるqualityが入っているので落としたdataframeを作る
df1 = df.drop(columns='quality')
y = df['quality'].values.reshape(-1,1)

scaler = preprocessing.StandardScaler()

# 重回帰
alpha = 10 ** (-4)
X = df1.values
X_fit = scaler.fit_transform(X)
df_ridge = pd.DataFrame()
df_lasso = pd.DataFrame()
while alpha <= 10 ** 6:
    model_ridge = linear_model.Ridge(alpha=alpha)
    model_ridge.fit(X_fit,y)
    tmp_se = pd.Series(np.append(model_ridge.coef_[0],alpha))
    df_ridge = df_ridge.append(tmp_se,ignore_index=True)

    model_lasso = linear_model.Lasso(alpha=alpha)
    model_lasso.fit(X_fit,y)
    tmp_se = pd.Series(np.append(model_lasso.coef_,alpha))
    df_lasso = df_lasso.append(tmp_se,ignore_index=True)
    alpha = alpha * 10 ** (0.1)
for column in df_ridge:
    if column != 11:
        plt.plot(df_ridge[11],df_ridge[column])
plt.xscale('log')
plt.gca().invert_xaxis()
plt.savefig("ridge.png")
plt.clf()
for column in df_lasso:
    if column != 11:
        plt.plot(df_lasso[11],df_lasso[column])
plt.xscale('log')
plt.gca().invert_xaxis()
plt.savefig("lasso.png")
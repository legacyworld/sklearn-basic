# 課題 4.2 リッジ回帰とラッソの正則化パス
# Youtubeでの解説：第5回(1) 15分50秒あたり
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing

#scikit-leanよりワインのデータをインポートする
df= pd.read_csv('winequality-red.csv',sep=';')
# 目標値であるqualityが入っているので落としたdataframeを作る
df1 = df.drop(columns='quality')
y = df['quality'].values.reshape(-1,1)
scaler = preprocessing.StandardScaler()
# 正則化パラメータ
alpha = 10 ** (-4)
X = df1.values
X_fit = scaler.fit_transform(X)
# 結果格納用のDataFrame
df_ridge = pd.DataFrame(columns=np.append(df1.columns,'alpha'))
df_lasso = pd.DataFrame(columns=np.append(df1.columns,'alpha'))
while alpha <= 10 ** 6 + 1:
    # リッジ回帰
    model_ridge = linear_model.Ridge(alpha=alpha)
    model_ridge.fit(X_fit,y)
    tmp_se = pd.Series(np.append(model_ridge.coef_[0],alpha),index=df_ridge.columns)
    df_ridge = df_ridge.append(tmp_se,ignore_index=True)
    # ラッソ回帰
    model_lasso = linear_model.Lasso(alpha=alpha)
    model_lasso.fit(X_fit,y)
    tmp_se = pd.Series(np.append(model_lasso.coef_,alpha),index=df_lasso.columns)
    df_lasso = df_lasso.append(tmp_se,ignore_index=True)
    alpha = alpha * 10 ** (0.1)

for column in df_ridge.drop(columns = 'alpha'):
    plt.plot(df_ridge['alpha'],df_ridge[column])
plt.xscale('log')
plt.gca().invert_xaxis()
plt.savefig("ridge.png")
plt.clf()
for column in df_lasso.drop(columns = 'alpha'):
    plt.plot(df_lasso['alpha'],df_lasso[column])
plt.xscale('log')
plt.gca().invert_xaxis()
plt.savefig("lasso.png")

plt.clf()
for column in df_ridge.drop(columns = 'alpha'):
    for index,value in df_ridge[column].iteritems():
        if index != 0 and abs(value) > abs(df_ridge[column][index-1]):
            plt.plot(df_ridge['alpha'],df_ridge[column],label=column)
            break

plt.plot(df_ridge['alpha'],df_ridge['residual sugar'])
plt.xscale('log')
plt.gca().invert_xaxis()
plt.legend()
plt.savefig("ridge_increase.png")

plt.clf()
for column in df_lasso.drop(columns = 'alpha'):
    for index,value in df_lasso[column].iteritems():
        if index != 0 and abs(value) > abs(df_lasso[column][index-1]):
            plt.plot(df_lasso['alpha'],df_lasso[column],label=column)
            break

plt.plot(df_lasso['alpha'],df_lasso['residual sugar'])
plt.xscale('log')
plt.gca().invert_xaxis()
plt.legend()
plt.savefig("lasso_increase.png")
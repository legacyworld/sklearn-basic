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
# 正則化パラメータ
alpha = 2 ** (-16)
X = df1.values
X_fit = scaler.fit_transform(X)
# 結果格納用のDataFrame
df_ridge_coeff = pd.DataFrame(columns=df1.columns)
df_ridge_result = pd.DataFrame(columns=['alpha','TrainErr','TestErr'])
df_lasso_coeff = pd.DataFrame(columns=df1.columns)
df_lasso_result = pd.DataFrame(columns=['alpha','TrainErr','TestErr'])
while alpha <= 2 ** 12:
    # リッジ回帰
    model_ridge = linear_model.Ridge(alpha=alpha)
    model_ridge.fit(X_fit,y)
    mse_ridge = mean_squared_error(model_ridge.predict(X_fit),y)
    scores_ridge = cross_val_score(model_ridge,X_fit,y,scoring="neg_mean_squared_error",cv=10)
    df_ridge_coeff = df_ridge_coeff.append(pd.Series(model_ridge.coef_[0],index=df_ridge_coeff.columns),ignore_index=True)
    df_ridge_result = df_ridge_result.append(pd.Series([alpha,mse_ridge,-scores_ridge.mean()],index=df_ridge_result.columns),ignore_index=True)    
    # ラッソ回帰
    model_lasso = linear_model.Lasso(alpha=alpha)
    model_lasso.fit(X_fit,y)
    mse_lasso = mean_squared_error(model_lasso.predict(X_fit),y)
    scores_lasso = cross_val_score(model_lasso,X_fit,y,scoring="neg_mean_squared_error",cv=10)
    df_lasso_coeff = df_lasso_coeff.append(pd.Series(model_lasso.coef_,index=df_lasso_coeff.columns),ignore_index=True)
    df_lasso_result = df_lasso_result.append(pd.Series([alpha,mse_lasso,-scores_lasso.mean()],index=df_lasso_result.columns),ignore_index=True)    
    alpha = alpha * 2

for index, row in df_ridge_coeff.iterrows():
    print(row.sort_values())
    print(df_ridge_result.iloc[index])
print(df_ridge_result.sort_values('TestErr'))

for index, row in df_lasso_coeff.iterrows():
    print(row.sort_values())
    print(df_lasso_result.iloc[index])
print(df_lasso_result.sort_values('TestErr'))
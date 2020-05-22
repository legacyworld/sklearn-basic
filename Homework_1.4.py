# 課題 1.4 wineデータの回帰
# Youtubeでの解説：第2回(1) 49分あたり
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

#scikit-leanよりワインのデータをインポートする
df= pd.read_csv('winequality-red.csv',sep=';')
# 目標値であるqualityが入っているので落としたdataframeを作る
df1 = df.drop(columns='quality')
y = df['quality'].values.reshape(-1,1)

scaler = preprocessing.StandardScaler()

# 単回帰　列ごとに行う
for column in df1:
    x = df[column]
    fig = plt.figure()
    plt.xlabel(column)
    plt.ylabel('quality')
    plt.scatter(x,y)
    # matrixへ変換
    X = x.values.reshape(-1,1)
    print(f"変換前は{x.values}")
    print(f"変換後は{X}")
    # スケーリング
    X_fit = scaler.fit_transform(X)
    model = linear_model.LinearRegression()
    model.fit(X_fit,y)
    plt.plot(x,model.predict(X_fit))
    mse = mean_squared_error(model.predict(X_fit),y)
    print(f"quality = {model.coef_[0][0]} * {column} + {model.intercept_[0]}")
    print(f"MSE: {mse}")
    filename = f"{column}.png"
    fig.savefig(filename)

# 重回帰
X = df1.values
X_fit = scaler.fit_transform(X)
model = linear_model.LinearRegression()
model.fit(X_fit,y)
print(model.coef_,model.intercept_)

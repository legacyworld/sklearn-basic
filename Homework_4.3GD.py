# 課題 4.3 最急降下法と確率的最急降下法
# Youtubeでの解説：第5回(1) 21分あたり
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate
import statsmodels.api as sm

class MyEstimator(BaseEstimator):
    def __init__(self,ep,eta,l):
        self.ep = ep
        self.eta = eta
        self.l = l
        self.loss = []
    # fit()を実装
    def fit(self, X, y):
        self.coef_ = self.grad_desc(X,y)
        # fit は self を返す
        return self

    # predict()を実装
    def predict(self, X):
        return np.dot(X, self.coef_)

    def grad_desc(self,X,y):
        m = len(y)
        loss = []
        diff = 10**(10)
        ep = self.ep
        # 特徴量の種類
        dim = X.shape[1]
        # betaの初期値
        beta = np.ones(dim).reshape(-1,1)
        eta = self.eta
        l = self.l
        while abs(diff) > ep:
            loss.append((1/(2*m))*np.sum(np.square(np.dot(X,beta)-y)))
            beta = beta*(1-2*l*eta) - eta*(1/m)*np.dot(X.T,(np.dot(X,beta)-y))
            if len(loss) > 1:
                diff = loss[len(loss)-1] - loss[len(loss)-2]
        self.loss = loss
        return beta

#scikit-leanよりワインのデータをインポートする
df= pd.read_csv('winequality-red.csv',sep=';')
# 目標値であるqualityが入っているので落としたdataframeを作る
df1 = df.drop(columns='quality')
y = df['quality'].values.reshape(-1,1)
X = df1.values
scaler = preprocessing.StandardScaler()
X_fit = scaler.fit_transform(X)
X_fit = sm.add_constant(X_fit) #最初の列に1を加える
epsilon = 10 ** (-7)
eta_list = [0.3,0.1,0.03]
loss = []
coef = []
for eta in eta_list:
    l = 10**(-5)
    test_min = 10**(9)
    while l <= 1/(2*eta):
        myest = MyEstimator(epsilon,eta,l)
        myest.fit(X_fit,y)
        scores = cross_validate(myest,X_fit,y,scoring="neg_mean_squared_error",cv=10)
        if abs(scores['test_score'].mean()) < test_min:
            test_min = abs(scores['test_score'].mean())
            loss = myest.loss
            l_min = l
            coef = myest.coef_
        l = l * 10**(0.5)
    plt.plot(loss)
    print(f"eta = {eta} : iter = {len(loss)}, loss = {loss[-1]}, lambda = {l_min}")
    # 係数の出力　一番最初に切片が入っているので2つ目から取り出して、最後に切片を出力
    i = 1
    for column in df1.columns:
        print(column,coef[i][0])
        i+=1
    print('intercept',coef[0][0])
plt.savefig("gd.png")

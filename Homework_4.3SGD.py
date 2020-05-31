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
        self.coef_ = self.stochastic_grad_desc(X,y)
        # fit は self を返す
        return self

    # predict()を実装
    def predict(self, X):
        return np.dot(X, self.coef_)

    def shuffle(self,X,y):
        r = np.random.permutation(len(y))
        return X[r],y[r]

    def stochastic_grad_desc(self,X,y):
        m = len(y)
        loss = []
        # 特徴量の種類
        dim = X.shape[1]
        # betaの初期値
        beta = np.ones(dim).reshape(-1,1)
        eta = self.eta
        l = self.l
        X_shuffle, y_shuffle = self.shuffle(X,y)
        # T回改善されなければ終了
        T = 100
        # 改善されない回数
        not_improve = 0
        # 目的関数最小値初期値
        min = 10 ** 9
        while True:
            for Xi,yi in zip(X_shuffle,y_shuffle):
                loss.append((1/(2*m))*np.sum(np.square(np.dot(X,beta)-y)))
                beta = beta*(1-2*l*eta) - eta*Xi.reshape(-1,1)*(np.dot(Xi,beta)-yi)
                if loss[len(loss)-1] < min:
                    min = loss[len(loss)-1]
                    min_beta = beta
                    not_improve = 0
                else:
                    # 目的関数の最小値が更新されない場合
                    not_improve += 1
                    if not_improve >= T:
                        break
            # 全サンプル終わったがT回以内に最小値が変わっている場合再度ループ
            if not_improve >= T:
                self.loss = loss
                break
        return min_beta

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
eta_list = [0.03,0.01,0.003]
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
    plt.plot(loss,label=f"$\eta$={eta}")
    print(f"eta = {eta} : iter = {len(loss)}, loss = {loss[-1]}, lambda = {l_min}, TestErr = {test_min}")
    # 係数の出力　一番最初に切片が入っているので2つ目から取り出して、最後に切片を出力
    i = 1
    for column in df1.columns:
        print(column,coef[i][0])
        i+=1
    print('intercept',coef[0][0])
plt.legend()
plt.savefig("sgd.png")

# 課題 7.3 ロジスティック回帰における勾配法とニュートン法の比較
# Youtubeでの解説：第8回(1) 27分あたり
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.base import BaseEstimator
import statsmodels.api as sm
from sklearn.datasets import load_iris  
iris = load_iris()

class MyEstimator(BaseEstimator):
    def __init__(self,ep,eta):
        self.ep = ep
        self.eta = eta
        self.loss = []

    def fit(self, X, y,f):
        m = len(y)
        loss = []
        diff = 10**(10)
        ep = self.ep
        # 特徴量の種類
        dim = X.T.shape[1]
        # betaの初期値
        beta = np.zeros(dim).reshape(-1,1)
        eta = self.eta
        
        while abs(diff) > ep:
            t_hat = self.sigmoid(beta.T,X)
            loss.append(-(1/m)*np.sum(y*np.log(t_hat) + (1-y)*np.log(1-t_hat)))
            # 再急降下法
            if f == "GD":
                beta = beta - eta*np.dot(X,(t_hat-y).reshape(-1,1))
            # ニュートン法
            else:
                # NxNの対角行列
                R = np.diag((t_hat*(1-t_hat))[0])
                # ヘッセ行列
                H = np.dot(np.dot(X,R),X.T)
                beta = beta - np.dot(np.linalg.inv(H),np.dot(X,(t_hat-y).reshape(-1,1)))
            if len(loss) > 1:
                diff = loss[len(loss)-1] - loss[len(loss)-2]
                if diff > 0:
                    break
        self.loss = loss
        self.coef_ = beta
        return self

    def sigmoid(self,w,x):
        return 1/(1+np.exp(-np.dot(w,x)))

# グラフ
fig = plt.figure(figsize=(20,10))
ax = [fig.add_subplot(3,3,i+1) for i in range(9)]

# virginicaかそうでないかだけ考慮する
target = 2
X = iris.data
y = iris.target
# y = 2ではない(virginicaではない)場合は0
y[np.where(np.not_equal(y,target))] = 0
y[np.where(np.equal(y,target))] = 1
scaler = preprocessing.StandardScaler()
X_fit = scaler.fit_transform(X)
X_fit = sm.add_constant(X_fit).T #最初の列に1を加える
epsilon = 10 ** (-8)
# 再急降下法
eta_list = [0.1,0.01,0.008,0.006,0.004,0.003,0.002,0.001,0.0005]
for index,eta in enumerate(eta_list):
    myest = MyEstimator(epsilon,eta)
    myest.fit(X_fit,y,"GD")
    ax[index].plot(myest.loss)
    ax[index].set_title(f"Optimization with Gradient Descent\nStepsize = {eta}\nIterations:{len(myest.loss)}; Initial Cost is:{myest.loss[0]:.3f}; Final Cost is:{myest.loss[-1]:.6f}")
    print(myest.coef_)
plt.tight_layout()    
plt.savefig(f"7.3GD.png")

# ニュートン法
myest.fit(X_fit,y,"newton")
plt.clf()
plt.plot(myest.loss) 
print(myest.coef_)   
plt.title(f"Optimization with Newton Method\nInitial Cost is:{myest.loss[0]:.3f}; Final Cost is:{myest.loss[-1]:.6f}")
plt.savefig("7.3Newton.png")
# 課題 5.3 ヒンジ損失と二乗損失
# Youtubeでの解説：第6回(1) 54分あたり
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import make_blobs

# 40個のランダムな分類データセット作成　centersで塊の数を指定
X, y = make_blobs(n_samples=40, centers=2, random_state=6)
# yの値を-1,1にする
y = y*2-1
# 二乗損失
clf = linear_model.LinearRegression(fit_intercept=True,normalize=True,copy_X=True)
clf.fit(X, y)
#分類データを描画。cmapの部分で色を勝手に決めてくれる
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# 決定境界の描画
x_plot = np.linspace(4,10,100)
w = [clf.intercept_,clf.coef_[0],clf.coef_[1]]
y_plot = -(w[1]/w[2]) * x_plot - w[0]/w[2]
plt.plot(x_plot,y_plot)
plt.savefig("5.3.png")

# 課題 5.3 ヒンジ損失と二乗損失
# Youtubeでの解説：第6回(1) 54分あたり
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# 40個のランダムな分類データセット作成　centersで塊の数を指定
X, y = make_blobs(n_samples=40, centers=2, random_state=6)
# kernel='linear'はヒンジ損失　Cが大きいほど正則化は効果が無くなる
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)
#分類データを描画。cmapの部分で色を勝手に決めてくれる
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# 決定境界の描画
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 30x30の格子を作る
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
# 各格子での分類
Z = clf.decision_function(xy).reshape(XX.shape)

# 等高線を使って決定境界を描画 level=0がそれにあたる
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# マージンが一番小さいサポートベクタを描画
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.savefig("5.3.png")
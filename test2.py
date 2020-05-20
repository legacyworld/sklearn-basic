#各種ライブラリのImport
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
from sklearn.preprocessing import PolynomialFeatures as PF
from sklearn import linear_model

NUM_TR = 6
np.random.seed(0)
rng = np.random.RandomState(0)

# 描画用のx軸データ
x_plot = np.linspace(0,10,100)
# 訓練データ
tmp = copy.deepcopy(x_plot)
rng.shuffle(tmp)
x_tr = np.sort(tmp[:NUM_TR])
y_tr = np.sin(x_tr) + 0.1*np.random.randn(NUM_TR)

# Matrixへ変換
X_tr = x_tr.reshape(-1,1)
X_plot = x_plot.reshape(-1,1)

# 多項式用のデータ
# 次数決め打ち
pf = PF(degree=5)
x_poly = pf.fit_transform(X_tr)
x_plot_poly = pf.fit_transform(X_plot)

model = linear_model.LinearRegression()
model.fit(x_poly,y_tr)

fig = plt.figure()
plt.scatter(x_tr,y_tr)
plt.plot(x_plot,model.predict(x_plot_poly))
plt.plot(x_plot,np.sin(x_plot))
fig.savefig('test.png')

# 多項式用のデータ
# 全ての次数
fig = plt.figure()
plt.scatter(x_tr,y_tr)

plt.plot(x_plot,np.sin(x_plot))

for degree in range(1,NUM_TR):
    pf = PF(degree=degree)
    x_poly = pf.fit_transform(X_tr)
    x_plot_poly = pf.fit_transform(X_plot)
    model = linear_model.LinearRegression()
    model.fit(x_poly,y_tr)
    plt.plot(x_plot,model.predict(x_plot_poly),label=f"degree {degree}")
    plt.legend()

plt.xlim(0,10)
plt.ylim(-2,2)
fig.savefig('test.png')
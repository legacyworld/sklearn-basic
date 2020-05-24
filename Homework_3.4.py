# 課題 3.2 3.4 多項式単回帰の正則化
# 第4回(1)　9分45秒あたり
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures as PF
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

DEGREE = 30

def true_f(x):
    return np.cos(1.5 * x * np.pi)

np.random.seed(0)
n_samples = 30

# 描画用のx軸データ
x_plot = np.linspace(0,1,100)
# 訓練データ
x_tr = np.sort(np.random.rand(n_samples))
y_tr = true_f(x_tr) + np.random.randn(n_samples) * 0.1
# Matrixへ変換
X_tr = x_tr.reshape(-1,1)
X_plot = x_plot.reshape(-1,1)
degree = DEGREE
alpha_list = [1e-30,1e-20,1e-10,1e-5, 1e-3,1e-2,1e-1, 1,10,100]
for alpha in alpha_list:
    plt.scatter(x_tr,y_tr,label="Training Samples")
    plt.plot(x_plot,true_f(x_plot),label="True")
    plt.xlim(0,1)
    plt.ylim(-2,2)
    filename = f"{alpha}.png"
    pf = PF(degree=degree,include_bias=False)
    linear_reg = linear_model.Ridge(alpha=alpha)
    steps = [("Polynomial_Features",pf),("Linear_Regression",linear_reg)]
    pipeline = Pipeline(steps=steps)
    pipeline.fit(X_tr,y_tr)
    plt.plot(x_plot,pipeline.predict(X_plot),label="Model")
    y_predict = pipeline.predict(X_tr)
    mse = mean_squared_error(y_tr,y_predict)
    scores = cross_val_score(pipeline,X_tr,y_tr,scoring="neg_mean_squared_error",cv=10)
    plt.title(f"Degree: {degree}, Lambda: {alpha}\nTrainErr: {mse:.2e} TestErr: {-scores.mean():.2e}(+/- {scores.std():.2e})")
    plt.legend()
    plt.savefig(filename)
    plt.clf()
    print(f"正則化パラメータ = {alpha}, 訓練誤差 = {mse}, テスト誤差 = {-scores.mean():.2e}")

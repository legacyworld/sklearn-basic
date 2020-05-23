# 課題 3.2 多項式単回帰の訓練誤差とテスト誤差
# 第4回(1) 40分あたり
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures as PF
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

DEGREE = 20

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

for degree in range(1,DEGREE+1):
    plt.scatter(x_tr,y_tr,label="Training Samples")
    plt.plot(x_plot,true_f(x_plot),label="True")
    plt.xlim(0,1)
    plt.ylim(-2,2)
    filename = f"{degree}.png"
    pf = PF(degree=degree)
    linear_reg = linear_model.LinearRegression()
    steps = [("Polynomial_Features",pf),("Linear_Regression",linear_reg)]
    pipeline = Pipeline(steps=steps)
    pipeline.fit(X_tr,y_tr)
    plt.plot(x_plot,pipeline.predict(X_plot),label="Model")
    y_predict = pipeline.predict(X_tr)
    mse = mean_squared_error(y_tr,y_predict)
    scores = cross_val_score(pipeline,X_tr,y_tr,scoring="neg_mean_squared_error",cv=10)
    plt.title(f"Degree: {degree} TrainErr: {mse:.2e} TestErr: {-scores.mean():.2e}(+/- {scores.std():.2e})")
    plt.legend()
    plt.savefig(filename)
    plt.clf()
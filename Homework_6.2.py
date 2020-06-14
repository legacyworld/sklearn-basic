# 課題 6.2 カーネルとSVM
# Youtubeでの解説：第7回(2) 48分30秒あたり
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
from sklearn import svm,metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles,make_moons,make_blobs

def fit_plot2(X,y,X_val,y_val,c_value,data_name,t):
    clf = svm.SVC(kernel='linear',C=c_value).fit(X,y)
    dec = clf.decision_function(X_val)
    predict = clf.predict(X_val)
    acc = metrics.accuracy_score(y_val,predict)
    plt.clf()
    # 決定境界の描画
    xlim = [-1.5,1.5]
    ylim = [-1.5,1.5]

    # 30x30の格子を作る
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    # 各格子での分類
    Z = clf.decision_function(xy).reshape(XX.shape)

    # 等高線を使って決定境界を描画 level=0がそれにあたる
    blue_rgb = mcolors.to_rgb("tab:blue")
    red_rgb = mcolors.to_rgb("tab:red")
    
    plt.contourf(XX, YY, Z,levels=[-2,-1,-0.5,0.5,1,2],colors=[red_rgb+(0.5,),red_rgb+(0.3,),(1,1,1),blue_rgb+(0.3,),blue_rgb+(0.5,)],extend='both')
    plt.contour(XX,YY,Z,levels=[0],linestyles=["--"])
    plt.scatter(X[:,0],X[:,1],c=y,edgecolors='k',cmap=ListedColormap(['#FF0000','#0000FF']))
    plt.title(f"{t} Accuracy = {acc}")
    plt.savefig(f"6.2_{data_name}_{t}.png")

def fit_plot(X,y,data_name):
    plt.clf()
    X = preprocessing.MinMaxScaler(feature_range=(-1,1)).fit_transform(X)
    plt.scatter(X[:,0],X[:,1],c=y,edgecolors='k',cmap=ListedColormap(['#FF0000','#0000FF']))
    plt.savefig(f"6.2_{data_name}.png")

    # Cを0.01から1,000まで変えた時のAUCスコア
    #c_values = [0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,100,1000]
    c_value = 0.01
    # TRAIN/VALIDATION/TEST用に3分割
    X_tr_val,X_test,y_tr_val,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
    X_tr,X_val,y_tr,y_val = train_test_split(X_tr_val,y_tr_val,test_size=0.2,random_state=42)
    fit_plot2(X_tr,y_tr,X_val,y_val,c_value,data_name,'Test')
    fit_plot2(X_tr_val,y_tr_val,X_test,y_test,c_value,data_name,'Training')
    print(len(y_tr_val),len(y_test),len(y_tr),len(y_val))

samples = 200
X,y = make_moons(n_samples=samples,noise=0.3,random_state=1)
fit_plot(X,y,'moons')
X,y = make_blobs(n_samples=samples,centers=2,random_state=64)
fit_plot(X,y,'linear_separation')
X,y = make_circles(n_samples=samples,noise=0.3,random_state=1)
fit_plot(X,y,'circles')
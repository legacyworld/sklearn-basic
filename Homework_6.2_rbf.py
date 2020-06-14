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

datanames = ['linear_separation','moons','circles']
samples = 200
c_values = [i/100 for i in range(1,30000,1000)]
# 3種類のデータ作成
def datasets(dataname):
    if dataname == 'linear_separation':
        X,y = make_blobs(n_samples=samples,centers=2,random_state=64)
    elif dataname == 'moons':
        X,y = make_moons(n_samples=samples,noise=0.3,random_state=74)
    elif dataname == 'circles':
        X,y = make_circles(n_samples=samples,noise=0.3,random_state=70)
    
    X = preprocessing.MinMaxScaler(feature_range=(-1,1)).fit_transform(X)
    return X,y
# 全てのCに対してaccuracy_scoreを返す関数
def learn_test_plot(clf_models):
    accs = {}
    for clf in clf_models:
        for dataname in datanames:
            if dataname not in accs:
                accs[dataname] = []
            X,y = datasets(dataname)
            X_tr_val,X_test,y_tr_val,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
            X_tr,X_val,y_tr,y_val = train_test_split(X_tr_val,y_tr_val,test_size=0.2,random_state=42)
            clf.fit(X_tr,y_tr)
            predict = clf.predict(X_val)
            train_acc = metrics.accuracy_score(y_val,predict)
            accs[dataname].append(train_acc)
    return accs
# gammaを動かして描画
gamma_list = [1000,100,1,0.1,0.01]
for gamma in gamma_list:
    plt.clf()
    clf_models = [svm.SVC(kernel='rbf',gamma=gamma,C=c_value) for c_value in c_values]
    accs = learn_test_plot(clf_models)
    fig = plt.figure(figsize=(20,10))
    ax = [fig.add_subplot(2,3,i+1) for i in range(6)]
    for a in ax:
        a.set_xlim(-1.5,1.5)
        a.set_ylim(-1.5,1.5)
    for dataname in datanames:
        best_c_index = np.argmax(accs[dataname])
        X,y = datasets(dataname)
        X_tr_val,X_test,y_tr_val,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
        X_tr,X_val,y_tr,y_val = train_test_split(X_tr_val,y_tr_val,test_size=0.2,random_state=42)
        clf = clf_models[best_c_index]
        
        clf.fit(X_tr,y_tr)
        train_predict = clf.predict(X_tr_val)
        test_predict = clf.predict(X_test)
        train_acc = metrics.accuracy_score(y_tr_val,train_predict) 
        test_acc = metrics.accuracy_score(y_test,test_predict)
        c_value = clf.get_params()['C']
        
        # メッシュデータ
        xlim = [-1.5,1.5]
        ylim = [-1.5,1.5]
        xx = np.linspace(xlim[0], xlim[1], 300)
        yy = np.linspace(ylim[0], ylim[1], 300)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = clf.decision_function(xy).reshape(XX.shape)
        # 塗りつぶし用の色
        blue_rgb = mcolors.to_rgb("tab:blue")
        red_rgb = mcolors.to_rgb("tab:red")
        # データセットごとに縦に並べる
        index = datanames.index(dataname)
        # decision_functionが大きいほど色を濃くする
        ax[index].contourf(XX, YY, Z,levels=[-2,-1,-0.1,0.1,1,2],colors=[red_rgb+(0.5,),red_rgb+(0.3,),(1,1,1),blue_rgb+(0.3,),blue_rgb+(0.5,)],extend='both')
        ax[index].contour(XX,YY,Z,levels=[0],linestyles=["--"])
        ax[index].scatter(X_tr_val[:,0],X_tr_val[:,1],c=y_tr_val,edgecolors='k',cmap=ListedColormap(['#FF0000','#0000FF']))
        ax[index].set_title(f"gamma = {gamma}\nTraining Accuracy = {train_acc} C = {c_value}")

        ax[index+3].contourf(XX, YY, Z,levels=[-2,-1,-0.1,0.1,1,2],colors=[red_rgb+(0.5,),red_rgb+(0.3,),(1,1,1),blue_rgb+(0.3,),blue_rgb+(0.5,)],extend='both')
        ax[index+3].contour(XX,YY,Z,levels=[0],linestyles=["--"])
        ax[index+3].scatter(X_test[:,0],X_test[:,1],c=y_test,edgecolors='k',cmap=ListedColormap(['#FF0000','#0000FF']))
        ax[index+3].set_title(f"gamma = {gamma}\nTest Accuracy = {test_acc} C = {c_value}")
    plt.savefig(f"6.2_{gamma}.png")
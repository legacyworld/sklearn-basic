# 課題 6.2 カーネルとSVM
# Youtubeでの解説：第7回(1) 25分50秒あたり
import numpy as np
import matplotlib.pyplot as plt
# シグモイド関数
def sigmoid(w,x):
    return 1/(1+np.exp(-np.dot(w,x)))

# 0.5で分類
def classification(a):
    return 1 if a > 0.5 else 0

X = np.array([[1.5,-0.5],[-0.5,-1.0],[1.0,-2.5],[1.5,-1.0],[0.5,0.0],[1.5,-2.0],[-0.5,-0.5],[1.0,-1.0],[0.0,-1.0],[0.0,0.5]])
# 切片部分が後ろに来ているので、1を最後に追加
X = np.concatenate([X,np.ones(10).reshape(-1,1)],1)
y = np.array([1,0,0,1,1,1,0,1,0,0])
w = np.array([[6,3,-2],[4.6,1,-2.2],[1,-1,-2]])
# 解説と同じ参考用のロジット等高線作成
fig = plt.figure(figsize=(20,10))
ax = [fig.add_subplot(2,2,i+1) for i in range(4)]
ax[0].scatter(X[:,0],X[:,1])
x_plot = np.linspace(-1.0,2.0,100)
ax[0].set_ylim(-3,1)
for i in range(0,3,1):
    y_plot = -w[i][2]/w[i][1]-w[i][0]/w[i][1]*x_plot
    ax[0].plot(x_plot,y_plot,label=f"w{i+1}")
ax[0].set_title("Sample Distribution")
ax[0].legend()
ax[0].grid(b=True)

# メッシュデータ
xlim = [-2.0,2.0]
ylim = [-3.0,3.0]
n = 100
xx = np.linspace(xlim[0], xlim[1], n)
yy = np.linspace(ylim[0], ylim[1], n)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel(),np.ones(n**2)])

for i in range(3):
    Z = sigmoid(w[i],xy).reshape(XX.shape)
    interval = np.arange(0,1,0.01)
    # 0が紫、1が赤、その間をグラデーション
    m = ax[i+1].contourf(XX,YY,Z,interval,cmap="rainbow",extend="both")
    m = ax[i+1].scatter(X[:,0],X[:,1],c=y)
    ax[i+1].set_title(f"w{i+1} Logit Contour")
    fig.colorbar(mappable = m,ax=ax[i+1])
plt.savefig("6.4.png")

# w^T x の計算
for index,w_i in enumerate(w):
    print(f"w{index+1} {np.dot(w_i,X.T)}")

# sigmoid(w^T x)の計算
np.set_printoptions(formatter={'float': '{:.2e}'.format})
for index,w_i in enumerate(w):
    print(f"w{index+1} {sigmoid(w_i,X.T)}")

# 分類
for index,w_i in enumerate(w):
    print(f"w{index+1} {np.vectorize(classification)(sigmoid(w_i,X.T))}")

# 確率
for index,w_i in enumerate(w):
    print(f"w{index+1} {np.count_nonzero(np.vectorize(classification)(sigmoid(w_i,X.T))==y)*10}%")
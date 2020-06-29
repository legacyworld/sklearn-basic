# 課題 8.7 主成分分析の例
# Youtubeでの解説：第9回(1) 30  分あたり
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris  
from sklearn.decomposition import PCA

iris = load_iris()
pca = PCA()
X = iris['data']
y = iris['target']
# 主成分分析
pca.fit(X)
transformed = pca.fit_transform(X)
# 寄与率
print(pca.explained_variance_ratio_)
# 描画
fig, ax = plt.subplots()
iris_dataframe = pd.DataFrame(transformed, columns=[0,1,2,3])
Axes = pd.plotting.scatter_matrix(iris_dataframe, c=y, figsize=(50, 50),ax=ax)
plt.savefig("8.7.png")

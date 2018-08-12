import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import style
import matplotlib.pyplot as plt

style.use("ggplot")
centers=[[1,1,1],[10,10,10],[5,5,5],[3,10,10]]
X,_=make_blobs(n_samples=100,centers=centers,cluster_std=1.4)

ms=MeanShift()
ms.fit(X)
labels=ms.labels_
cluster_centers=ms.cluster_centers_
print(cluster_centers)
n_clusters=len(np.unique(labels))
print(n_clusters)
colors=10*['r','g','b','c','k','y','m']
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
for i in range(len(X)):
    ax.scatter(X[i][0],X[i][1],X[i][2],c=colors[labels[i]],marker='o')

ax.scatter(cluster_centers[:,0],cluster_centers[:,1],cluster_centers[:,2],marker='x',color='k',s=150,linewidths=5,zorder=10)
plt.show()
# DBSCAN Clustering

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"C:\Users\Asus\OneDrive - IIT Kanpur\Desktop\ml_data\mall_data.csv")
X = dataset.iloc[:, [3, 4]].values


# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import DBSCAN,KMeans,AgglomerativeClustering
dbscan=DBSCAN(eps=3,min_samples=4)
dbscan1=DBSCAN(eps=4,min_samples=4)
kmeans = KMeans(n_clusters=2, random_state=0)
aglo=AgglomerativeClustering(n_clusters= 2,affinity= "euclidean")

# Fitting the model

model=dbscan.fit(X)
model1=dbscan1.fit(X)
model2=kmeans.fit(X)
model3=aglo.fit(X)


labels=model.labels_
labels1=model1.labels_
labels2=model2.labels_
labels3=model3.labels_

from sklearn import metrics

#identifying the points which makes up our core points
sample_cores=np.zeros_like(labels,dtype=bool)
sample_cores1=np.zeros_like(labels1,dtype=bool)
sample_cores2=np.zeros_like(labels2,dtype=bool)
sample_cores3=np.zeros_like(labels3,dtype=bool)

sample_cores[dbscan.core_sample_indices_]=True
sample_cores[dbscan1.core_sample_indices_]=True
#sample_cores1[KMeans.core_sample_indices_]=True

#Calculating the number of clusters

n_clusters=len(set(labels))- (1 if -1 in labels else 0)
n_clusters1=len(set(labels1))- (1 if -1 in labels1 else 0)
n_clusters2=len(set(labels2))- (1 if -1 in labels2 else 0)
n_clusters3=len(set(labels3))- (1 if -1 in labels2 else 0)




print(metrics.silhouette_score(X,labels))
print(metrics.silhouette_score(X,labels1))
print(metrics.silhouette_score(X,labels2))
print(metrics.silhouette_score(X,labels3))




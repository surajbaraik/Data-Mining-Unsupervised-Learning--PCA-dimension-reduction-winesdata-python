# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:45:56 2020

@author: suraj baraik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

wine=pd.read_csv("C:\\Users\\suraj baraik\\Desktop\\Data Science\\Suraj\\New folder (12)\\Module 14 Data Mining Unsupervised Learning- Dimension Reduction PCA)\\wine.csv")
wine.head()
wine.describe()

# Considering only numerical data 
wine.drop(['Type'],axis=1,inplace=True)
wine.describe()

# Normalizing the numerical data 
wine_normal=scale(wine)
wine_normal=pd.DataFrame(wine_normal) ##Converting from float to Dataframe format 

pca=PCA(n_components=13)
pca_values=pca.fit_transform(wine_normal)

var=pca.explained_variance_ratio_
var

pca.components_[0]
pca.components_

var1=np.cumsum(np.round(var,decimals=4)*100)
var1

plt.plot(var1,color='red')

################### Clustering  ##########################
new_def=pd.DataFrame(pca_values[:,0:3])
new_def.head()
new_def.describe

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch ##for creating dendrogram

type(new_def)

z=linkage(new_def,method="complete",metric="euclidean")

plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 5 as clusters from the dendrogram
from sklearn.cluster import AgglomerativeClustering
h_complete	=	AgglomerativeClustering(n_clusters=5,linkage='complete',affinity = "euclidean").fit(new_def) 

h_complete.labels_

cluster_labels=pd.Series(h_complete.labels_)
wine['clust']=cluster_labels # creating a  new column and assigning it to new column 
wine = pd.concat([cluster_labels,wine],axis=1)

# getting aggregate mean of each cluster
wine.groupby(wine.clust).mean()

# creating a csv file 
wine.to_csv("winehie.csv",index=False) #,encoding="utf-8")

import os
os.getcwd()

################### KMean-Clustering  ##########################
new_df = pd.DataFrame(pca_values[:,0:3])

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

###### screw plot or elbow curve ############
k = list(range(2,15))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(new_def)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(new_def.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,new_def.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=3) 
model.fit(new_def)

model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 

wine['clust']=md # creating a  new column and assigning it to new column 
new_def.head()

wine.groupby(wine.clust).mean()
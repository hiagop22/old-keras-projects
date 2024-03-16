#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Created on 04-10-2020 / 18:57:31
    @author: Hiago dos Santos
    @e-mail: hiagop22@gmail.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("dataset.csv")
df.columns = ["CustomerID", "Gender", "Age", "Annual_Income", "Spending_Score"] # Define the titles
print(df.head(5))
# print(df.shape)
# print(df.isnull().sum()) # verify if there's some zero value

x = df.iloc[:, [3, 4]]

scaler = StandardScaler()
x = scaler.fit_transform(x)
# print(x)

df.iloc[:, [3,4]] = x
# plt.scatter(x[:,0], x[:,1]) # discret samples
# plt.show() 

# Elbow method
sse = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(x)
    sse.append(kmeans.inertia_)

# plt.title('Elbow')
# plt.xlabel('number of clusters')
# plt.ylabel('Quadratic Square Error')
# plt.plot(range(1, 11), sse) # plt pois quero fazer uma interpolaçao de forma "contínua"
# plt.show()

n_clusters = 5
model = KMeans(n_clusters=n_clusters, init='k-means++')
pred = model.fit_predict(x)

plt.figure(figsize=(20,10))
for i in range(n_clusters):
    plt.scatter(x[pred==i,0], x[pred==i,1], s=50, label='Cluster %d' %i)

plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], 
            s =100, c='black', label='Centroids')

plt.title("Clusters")
plt.xlabel("Annual Income")
plt.ylabel("Spending_Score")
plt.legend()
plt.show()

# the number of cluster  isn't fixed in a sample, look at the cluster number and 
# give it for the program
selected_cluster = int(input("Number of cluster (0,1,2,3,4): "))
                                
np.savetxt('ids.txt', df.iloc[pred == selected_cluster, 0], fmt='%s')
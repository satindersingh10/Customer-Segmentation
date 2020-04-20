# K Means Clustering

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataSet = pd.read_csv('F:\CSE\Machine Learning\My ML Dir\Part 4 - Clustering\Section 24 - K-Means Clustering\Mall_Customers.csv')
X = dataSet.iloc[:,[3,4]].values

# using elbow method to find no. of clusters K
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('Elbow Method')
plt.show()

# applying kMeans to mall dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans =  kmeans.fit_predict(X)

# Visualizing the cluster
plt.scatter(X[y_kmeans ==0, 0], X[y_kmeans ==0 , 1], s=100, color = 'red', label = 'CAREFUL')
plt.scatter(X[y_kmeans ==1, 0], X[y_kmeans ==1 , 1], s=100, color = 'blue', label = 'STANDARD')
plt.scatter(X[y_kmeans ==2, 0], X[y_kmeans ==2 , 1], s=100, color = 'green', label = 'TARGET')
plt.scatter(X[y_kmeans ==3, 0], X[y_kmeans ==3 , 1], s=100, color = 'yellow', label = 'CARELESS')
plt.scatter(X[y_kmeans ==4, 0], X[y_kmeans ==4 , 1], s=100, color = 'cyan', label = 'SENSIBLE')
#plotting centroid
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s= 200, color ='black', label = 'CENTROID')
plt.title('Clusters of Client')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

print("SOLUTION TO TSF_GRIP21_DS_TASK2 BY PRANJAL KALEKAR")
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets

#loading data
iris = datasets.load_iris()
iris_data = pd.DataFrame(iris.data, columns= iris.feature_names)
data = iris.data
print(iris_data.head())


#to apply kmean we must have sutable value of k for that need to apply elbo method on 'sse'\
#sum of squared errors and numbersfrom 1-10

sse = []
clust = range(1,11)
for i in clust:
    kmean = KMeans(n_clusters = i, random_state = 0)
    kmean.fit(data)
    sse.append(kmean.inertia_)
print("Sum of squared errors :", sse)

plt.figure(figsize = (8,6))
plt.plot(clust, sse)
plt.xlabel("K values") 
plt.ylabel("SSE")
plt.show()   

print("From above we can conclude that 3 is the most suitable value for number of clusters")


#creating a Kmean classifier
kmean = KMeans(n_clusters = 3, random_state = 0)
data_mean = kmean.fit_predict(data)
print("Mean of data by kmean classifier")
print(data_mean)

#displaying data and visualysing clusters
plt.figure(figsize = (8,6))
plt.scatter(data[data_mean ==0,0], data[data_mean ==0,1], c = 'red', label = 'setosa', marker='+')

plt.scatter(data[data_mean ==1,0], data[data_mean ==1,1], c = 'blue', label = 'versicolour', marker = '+')

plt.scatter(data[data_mean ==2,0], data[data_mean ==2,1], c = 'green', label = 'virginica', marker = '+')

plt.scatter(kmean.cluster_centers_[:,0], kmean.cluster_centers_[:,1] ,  c ='yellow', label = "Centroid")

plt.legend()
# 1 - Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats

# 2 - Import Data
data = pd.read_csv('Mall_Customers.csv')
data.head()

# 3 - Amati Bentuk Data
data.shape

# 4 - Melihat Ringkasan Statistik Deskriptif dari Dataframe
df = pd.DataFrame(data) 
print(df)

# 5 - Cek Null Data
print(df.isnull().sum())

# 6 - Cek Outlier
kolom_numerik = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'] 
z_scores = np.abs(stats.zscore(df[kolom_numerik])) 

# 7 - Tangani Outlier dengan Nilai Mean
outliers_z = df[(z_scores > 3).any(axis=1)] 
print("Jumlah outlier berdasarkan Z-score:", len(outliers_z))
print("Outlier berdasarkan Z-score:\n", outliers_z)

# 8 - Amati Bentuk Visual Masing-Masing Fitur
plt.style.use('fivethirtyeight')
plt.figure(1 , figsize = (15 , 6))
n = 0
for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1 , 3 , n)
    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)
    sns.histplot(
        df[x],
        kde=True,
        stat="density",
        kde_kws=dict(cut=3),
        bins = 20
        )
    plt.title('Distplot of {}'.format(x))
plt.show()

# 9 - Ploting untuk Mencari Relasi antara Age , Annual Income dan Spending Score
plt.figure(1 , figsize = (15 , 20))
n = 0
for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
    for y in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
        n += 1
        plt.subplot(3 , 3 , n)
        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
        sns.regplot(x = x , y = y , data = df)
        plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )
plt.show()

# 10 - Melihat Sebaran Spending Score dan Annual Income pada Gender
plt.figure(figsize=(10, 6))
for gender in ["Male", "Female"]:
    subset = df[df["Gender"] == gender]
    plt.scatter(
        subset["Annual Income (k$)"],
        subset["Spending Score (1-100)"],
        s=200,
        alpha=0.5,
        label=gender
        )
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Annual Income vs Spending Score by Gender")
plt.legend()
plt.show()

# 11 - Merancang K-Means untuk Spending Score vs Annual Income
# Menentukan nilai k yang sesuai dengan Elbow-Method
X1 = df[['Annual Income (k$)' , 'Spending Score (1-100)']].iloc[: , :].values
inertia = []
for n in range(1 , 11):
    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10,max_iter=300, random_state= 111) )
    algorithm.fit(X1)
    inertia.append(algorithm.inertia_)

# Plot bentuk visual elbow
plt.figure(1 , figsize = (15 ,6))
plt.plot(range(1 , 11) , inertia , 'o')
plt.plot(range(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()

# 12 - Membangun K-Means
algorithm = (KMeans(n_clusters = 5, init='k-means++', n_init = 10, max_iter=300, tol=0.0001, random_state= 111, algorithm='elkan'))
algorithm.fit(X1)
labels2 = algorithm.labels_
centroids2 = algorithm.cluster_centers_

# 13 - Menyiapkan Data untuk Bentuk Visual Cluster
labels2 = algorithm.labels_
centroids2 = algorithm.cluster_centers_
step = 0.02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step),np.arange(y_min, y_max, step))
Z1 = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) # array diratakan 1D

# 14 - Melihat Bentuk Visual Cluster
plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z1 = Z1.reshape(xx.shape)
plt.imshow(
    Z1,
    interpolation='nearest',
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap = plt.cm.Pastel2,
    aspect = 'auto',
    origin='lower'
    )
plt.scatter(
    x = 'Annual Income (k$)',
    y = 'Spending Score (1-100)',
    data= df ,
    c = labels2,
    s = 200
    )
plt.scatter(
    x = centroids2[: , 0],
    y = centroids2[: , 1],
    s = 300 ,
    c = 'red',
    alpha = 0.5
    )
plt.ylabel('Spending Score (1-100)')
plt.xlabel('Annual Income(k$)')
plt.show()

# 15 - Melihat Nilai Silhouette Score
score2 = silhouette_score(X1, labels2)
print("Silhouette Score: ", score2)

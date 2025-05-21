# 1 - Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import skfuzzy as fuzz

# 2 - Import Data
data = pd.read_csv('Mall_Customers.csv')

# 3 - Amati Bentuk Data
data.shape

# 4 - Melihat Ringkasan Statistik Deskriptif dari Dataframe
df = pd.DataFrame(data)
print(df)

# 5 - Cek Null Data
print(df.isnull().sum())

# 6 - Tangani Outlier
kolom_numerik = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'] 
z_scores = np.abs(stats.zscore(data[kolom_numerik])) 

# 7 - Tangani Outlier dengan Nilai Mean
outliers_z = df[(z_scores > 3).any(axis=1)]
print("Jumlah outlier berdasarkan Z-score:", len(outliers_z))
print("Outlier berdasarkan Z-score:\n", outliers_z)

# 8 - Amati Bentuk Visual Masing-Masing Fitur
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15, 6))
for i, col in enumerate(kolom_numerik):
    plt.subplot(1, 3, i+1)
    sns.histplot(data[col], kde=True, bins=20)
    plt.title(f'Distribution of {col}')
plt.show()

# 9 - Ploting untuk Mencari Relasi antara Age , Annual Income dan Spending Score
plt.figure(figsize=(15, 20))
plot_no = 1
for x in kolom_numerik:
    for y in kolom_numerik:
        plt.subplot(3, 3, plot_no)
        sns.regplot(data=data, x=x, y=y)
        plot_no += 1
plt.show()

# 10 - Melihat Sebaran Spending Score dan Annual Income pada Gender
plt.figure(figsize=(10, 6))
for gender in ['Male', 'Female']:
    subset = data[data['Gender'] == gender]
    plt.scatter(subset['Annual Income (k$)'], subset['Spending Score (1-100)'], s=200, alpha=0.5, label=gender)
plt.title("Annual Income vs Spending Score by Gender")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

# 11 - Merancang K-Means untuk Spending Score vs Annual Income
X1 = data[['Annual Income (k$)', 'Spending Score (1-100)']].values

inertia = []
for n in range(1, 11):
    kmeans = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300, random_state=111)
    kmeans.fit(X1)
    inertia.append(kmeans.inertia_)

# Elbow Method Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, 'o-')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Final K-Means with 5 clusters
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=111)
kmeans.fit(X1)
labels_kmeans = kmeans.labels_
centroids_kmeans = kmeans.cluster_centers_

# Plot K-Means Clusters
plt.figure(figsize=(10, 6))
plt.scatter(X1[:, 0], X1[:, 1], c=labels_kmeans, s=100, cmap='viridis')
plt.scatter(centroids_kmeans[:, 0], centroids_kmeans[:, 1], c='red', s=300, alpha=0.5)
plt.title('K-Means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# Silhouette Score - KMeans
score_kmeans = silhouette_score(X1, labels_kmeans)
print("K-Means Silhouette Score:", score_kmeans)

# 12 - Merancang Fuzzy C-Means Clustering untuk Spending Score vs Annual Income 
X1_T = X1.T
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(X1_T, c=5, m=2, error=0.005, maxiter=1000)
labels_fcm = np.argmax(u, axis=0)

# Plot FCM
plt.figure(figsize=(10, 6))
plt.scatter(X1[:, 0], X1[:, 1], c=labels_fcm, s=100, cmap='Accent')
plt.scatter(cntr[:, 0], cntr[:, 1], c='red', s=300, alpha=0.5)
plt.title('Fuzzy C-Means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# Silhouette Score - FCM
score_fcm = silhouette_score(X1, labels_fcm)
print("Fuzzy C-Means Silhouette Score:", score_fcm)

# 13 - DBSCAN Clustering
scaler = StandardScaler()
X1_scaled = scaler.fit_transform(X1)

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X1_scaled)

# Plot DBSCAN
plt.figure(figsize=(10, 6))
unique_labels = set(labels_dbscan)
colors = sns.color_palette('tab10', len(unique_labels))

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = 'black'
    class_member_mask = (labels_dbscan == k)
    xy = X1[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=10)

plt.title('DBSCAN Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# Silhouette Score - DBSCAN
if len(set(labels_dbscan)) > 1 and -1 in labels_dbscan:
    score_dbscan = silhouette_score(X1_scaled[labels_dbscan != -1], labels_dbscan[labels_dbscan != -1])
    print("DBSCAN Silhouette Score (tanpa noise):", score_dbscan)
elif len(set(labels_dbscan)) > 1:
    score_dbscan = silhouette_score(X1_scaled, labels_dbscan)
    print("DBSCAN Silhouette Score:", score_dbscan)
else:
    score_dbscan = None
    print("DBSCAN tidak menemukan cluster yang valid.")

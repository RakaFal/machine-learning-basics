import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Dataset
df = pd.read_csv('stroke.csv')

# Deskripsi Data
print(df.info())
print(df.describe())

# PreProcessing
# 1. Cek Missing Value
df = df[df['gender'] != 'Other']
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
df['smoking_status'] = df['smoking_status'].fillna('Unknown')

# 2. Visualisasi distribusi data
# BarPlot for Gender
plt.hist(df['gender'], bins=3, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.title('Histogram of Gender')
plt.show()

# Histogram for Age (continuous numeric column)
plt.hist(df['age'], bins=10, color='purple', edgecolor='black', alpha=0.7)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram of Age')
plt.show()

# Histogram for Hypertension (binary categorical column)
plt.hist(df['hypertension'], bins=2, color='orange', edgecolor='black', alpha=0.7)
plt.xlabel('Hypertension')
plt.ylabel('Frequency')
plt.title('Histogram of Hypertension')
plt.show()

# Histogram for Heart Disease (binary categorical column)
plt.hist(df['heart_disease'], bins=2, color='red', edgecolor='black', alpha=0.7)
plt.xlabel('Heart Disease')
plt.ylabel('Frequency')
plt.title('Histogram of Heart Disease')
plt.show()

# Histogram for Ever Married (target binary column)
plt.hist(df['ever_married'], bins=3, color='pink', edgecolor='black', alpha=0.7)
plt.xlabel('Ever Married')
plt.ylabel('Frequency')
plt.title('Histogram of Ever Married')
plt.show()

# Histogram for Work Type (categorical column)
plt.hist(df['work_type'], bins=10, color='blue', edgecolor='black')
plt.xlabel('Work Type')
plt.ylabel('Frequency')
plt.title('Histogram of Work Type')
plt.show()

# Histogram for Residence Type (categorical column)
plt.hist(df['Residence_type'], bins=3, color='cyan', edgecolor='black', alpha=0.7)
plt.xlabel('Residence Type')
plt.ylabel('Frequency')
plt.title('Histogram of Residence Type')
plt.show()

# Histogram for Average Glucose Level (continuous numeric column)
plt.hist(df['avg_glucose_level'], bins=8, color='green', edgecolor='black', alpha=0.7)
plt.xlabel('Average Glucose Level')
plt.ylabel('Frequency')
plt.title('Histogram of Average Glucose Level')
plt.show()

# Histogram for BMI (continuous numeric column)
plt.hist(df['bmi'], bins=6, color='purple', edgecolor='black', alpha=0.7)
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.title('Histogram of BMI')
plt.show()

# Histogram for Smoking Status (categorical column)
plt.hist(df['smoking_status'], bins=8, color='cyan', edgecolor='black', alpha=0.7)
plt.xlabel('Smoking Status')
plt.ylabel('Frequency')
plt.title('Histogram of Smoking Status')
plt.show()

# Histogram for Stroke (target binary column)
plt.hist(df['stroke'], bins=2, color='brown', edgecolor='black', alpha=0.7)
plt.xlabel('Stroke')
plt.ylabel('Frequency')
plt.title('Histogram of Stroke (Target Variable)')
plt.show()

# 3. Konversi Variabel Kategorikal
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# 4. Deteksi outlier dengan Z-Score
print("\nDeteksi outlier dengan Z-Score:")
numerical_columns = ['age', 'avg_glucose_level', 'bmi']
z_scores = np.abs(stats.zscore(df[numerical_columns]))
outliers_z = df[(z_scores > 3).any(axis=1)]
print("\nOutlier berdasarkan Z-score:\n", outliers_z)


# 5. Menghapus outlier (Z-Score)
print("\nMenghapus outlier:")
z_scores = np.abs(stats.zscore(df[numerical_columns]))
outliers_z = df[(z_scores > 3).any(axis=1)]
print("\nOutlier berdasarkan Z-score:\n", outliers_z)
df_cleaned = df[(z_scores <= 3).all(axis=1)]
print("\nDataset sesudah menghapus outlier:\n", df_cleaned.head())

# 6. Mengganti outlier dengan mean atau median
print("\nMengganti outlier dengan mean atau median:")
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
z_scores = df[numerical_columns].apply(lambda x: stats.zscore(x, nan_policy='omit'))  # nan_policy='omit' agar mengabaikan NaN

outliers_z = df[(z_scores > 3).any(axis=1)]
print("\nOutlier berdasarkan Z-score:\n", outliers_z)

# Pearson
# Menghitung korelasi Pearson
corr_matrix = df.corr()

# Menampilkan heatmap korelasi
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriks Korelasi Pearson')
plt.show()

# Simpan hasil PCA ke CSV
corr_matrix.to_csv("pearson_results.csv", index=False)
print("Hasil Pearson telah disimpan dalam file pearson_results.csv")

# Normalisasi fitur numerik
scaler = MinMaxScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Pilih fitur (X) dan target (y)
X = df.drop(columns=['id', 'stroke'])  # Hapus ID & Target
y = df['stroke']

# Terapkan SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Cek distribusi sebelum dan sesudah SMOTE
# --- Distribusi Sesudah SMOTE ---
print("Distribusi Awal:")
print(y.value_counts())
plt.figure(figsize=(6,4))
sns.barplot(x=y.value_counts().index, 
            y=y.value_counts().values,
            hue=y.value_counts().index, 
            palette='bright',
            legend=False)
plt.xticks(ticks=[0,1], labels=['Sehat', 'Stroke'])
plt.xlabel("Status")
plt.ylabel("Total")
plt.title("Distribusi Awal")
plt.show()

# --- Distribusi Sesudah SMOTE ---
print("\nDistribusi Sesudah SMOTE:")
print(pd.Series(y_resampled).value_counts())
plt.figure(figsize=(6,4))
sns.barplot(x=pd.Series(y_resampled).value_counts().index,
            y=pd.Series(y_resampled).value_counts().values,
            hue=pd.Series(y_resampled).value_counts().index, 
            palette='bright',
            legend=False)
plt.xticks(ticks=[0,1], labels=['Sehat', 'Stroke'])
plt.xlabel("Status")
plt.ylabel("Total")
plt.title("Distribusi Sesudah SMOTE")
plt.show()

print("\nTotal Data Sebelum SMOTE:")
print(X, y)

print("\nTotal Data Sesudah SMOTE:")
print(X_resampled, y_resampled)

# Simpan hasil sesudah SMOTE ke dalam file CSV
resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
resampled_df['stroke'] = y_resampled
resampled_df.to_csv("pearson_oversampling.csv", index=False)
print("Hasil SMOTE telah disimpan dalam 'pearson_oversampling.csv'")

# Terapkan RandomUnderSampler
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)

# Cek distribusi sesudah undersampling
print("\nDistribusi Sesudah Undersampling:")
print(pd.Series(y_resampled).value_counts())

# Plot distribusi sesudah undersampling
plt.figure(figsize=(6,4))
sns.barplot(x=pd.Series(y_resampled).value_counts().index,
            y=pd.Series(y_resampled).value_counts().values,
            hue=y.value_counts().index,
            palette='bright',
            legend=False)
plt.xticks(ticks=[0,1], labels=['Sehat', 'Stroke'])
plt.xlabel("Status")
plt.ylabel("Total")
plt.title("Distribusi Sesudah Undersampling")
plt.show()

print("\nTotal Data Sebelum UnderSampling:")
print(X, y)

print("\nTotal Data Sesudah UnderSampling:")
print(X_resampled, y_resampled)

# Simpan hasil sesudah Random UnderSampling ke dalam file CSV
under_df = pd.DataFrame(X_resampled, columns=X.columns)
under_df['stroke'] = y_resampled
under_df.to_csv("pearson_undersampling.csv", index=False)
print("Hasil Random UnderSampling telah disimpan dalam 'pearson_undersampling.csv'")

# PCA
# Memisahkan fitur dan target
X = df.drop(columns=["id", "stroke"])  # Hanya fitur numerik
y = df["stroke"]  # Target

# Standarisasi data (PCA membutuhkan data dalam skala yang sama)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Melakukan PCA (jumlah komponen bisa dikurangi)
pca = PCA(n_components=2)  # Mengambil 2 komponen utama untuk visualisasi
X_pca = pca.fit_transform(X_scaled)

# Mengubah hasil PCA ke DataFrame
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["stroke"] = y  # Tambahkan label kembali untuk visualisasi

# Menampilkan varians yang dijelaskan oleh masing-masing komponen utama
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Explained:", sum(pca.explained_variance_ratio_))

# Plot hasil PCA
plt.figure(figsize=(8, 6))
plt.scatter(df_pca["PC1"], df_pca["PC2"], c=df_pca["stroke"], cmap="rainbow", alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA: Stroke Dataset")
plt.colorbar(label="stroke")
plt.show()

# Simpan hasil PCA ke CSV
df_pca.to_csv("pca_results.csv", index=False)
print("Hasil PCA telah disimpan dalam file pca_results.csv")

# Pilih fitur (X) dan target (y)
X = df.drop(columns=['id', 'stroke'])  # Hapus ID & Target
y = df['stroke']

# Terapkan SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Cek distribusi sebelum dan sesudah SMOTE
# --- Distribusi Sesudah SMOTE ---
print("Distribusi Awal:")
print(y.value_counts())
plt.figure(figsize=(6,4))
sns.barplot(x=y.value_counts().index, 
            y=y.value_counts().values,
            hue=y.value_counts().index, 
            palette='bright',
            legend=False)
plt.xticks(ticks=[0,1], labels=['Sehat', 'Stroke'])
plt.xlabel("Status")
plt.ylabel("Total")
plt.title("Distribusi Sebelum SMOTE")
plt.show()

# --- Distribusi Sesudah SMOTE ---
print("\nDistribusi Sesudah SMOTE:")
print(pd.Series(y_resampled).value_counts())
plt.figure(figsize=(6,4))
sns.barplot(x=pd.Series(y_resampled).value_counts().index,
            y=pd.Series(y_resampled).value_counts().values,
            hue=pd.Series(y_resampled).value_counts().index, 
            palette='bright',
            legend=False)
plt.xticks(ticks=[0,1], labels=['Sehat', 'Stroke'])
plt.xlabel("Status")
plt.ylabel("Total")
plt.title("Distribusi Sesudah SMOTE")
plt.show()

print("\nTotal Data Sebelum SMOTE:")
print(X, y)

print("\nTotal Data Sesudah SMOTE:")
print(X_resampled, y_resampled)

# Simpan hasil sesudah SMOTE ke dalam file CSV
resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
resampled_df['stroke'] = y_resampled
resampled_df.to_csv("pca_oversampling.csv", index=False)
print("Hasil SMOTE telah disimpan dalam 'pca_oversampling.csv'")

# Terapkan RandomUnderSampler
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)

# Cek distribusi sesudah undersampling
print("\nDistribusi Sesudah Undersampling:")
print(pd.Series(y_resampled).value_counts())

# Plot distribusi sesudah undersampling
plt.figure(figsize=(6,4))
sns.barplot(x=pd.Series(y_resampled).value_counts().index,
            y=pd.Series(y_resampled).value_counts().values,
            hue=y.value_counts().index,
            palette='bright',
            legend=False)
plt.xticks(ticks=[0,1], labels=['Sehat', 'Stroke'])
plt.xlabel("Status")
plt.ylabel("Total")
plt.title("Distribusi Sesudah Undersampling")
plt.show()

print("\nTotal Data Sebelum UnderSampling:")
print(X, y)

print("\nTotal Data Sesudah UnderSampling:")
print(X_resampled, y_resampled)

# Simpan hasil sesudah Random UnderSampling ke dalam file CSV
under_df = pd.DataFrame(X_resampled, columns=X.columns)
under_df['stroke'] = y_resampled
under_df.to_csv("pca_undersampling.csv", index=False)
print("Hasil Random UnderSampling telah disimpan dalam 'pca_undersampling.csv'")

# t-SNE
# Terapkan t-SNE (gunakan n_components=2 untuk visualisasi 2D)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Mengubah hasil t-SNE ke DataFrame
df_tsne = pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2"])
df_tsne["stroke"] = y  # Tambahkan label kembali untuk visualisasi

# Simpan hasil t-SNE ke CSV
df_tsne.to_csv("tsne_results.csv", index=False)
print("Hasil t-SNE telah disimpan dalam file tsne_results.csv")

# Menampilkan visualisasi t-SNE
plt.figure(figsize=(8, 6))
sns.scatterplot(x="TSNE1", y="TSNE2", hue="stroke", palette="coolwarm", data=df_tsne, alpha=0.7, s=100)
plt.title("t-SNE: Stroke Dataset")
plt.legend(title="Stroke", loc="best")
plt.show()

# Pilih fitur (X) dan target (y)
X = df.drop(columns=['id', 'stroke'])  # Hapus ID & Target
y = df['stroke']

# Terapkan SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Cek distribusi sebelum dan sesudah SMOTE
# --- Distribusi Sesudah SMOTE ---
print("Distribusi Sebelum SMOTE:")
print(y.value_counts())
plt.figure(figsize=(6,4))
sns.barplot(x=y.value_counts().index, 
            y=y.value_counts().values,
            hue=y.value_counts().index, 
            palette='bright',
            legend=False)
plt.xticks(ticks=[0,1], labels=['Sehat', 'Stroke'])
plt.xlabel("Status")
plt.ylabel("Total")
plt.title("Distribusi Sebelum SMOTE")
plt.show()

# --- Distribusi Sesudah SMOTE ---
print("\nDistribusi Sesudah SMOTE:")
print(pd.Series(y_resampled).value_counts())
plt.figure(figsize=(6,4))
sns.barplot(x=pd.Series(y_resampled).value_counts().index,
            y=pd.Series(y_resampled).value_counts().values,
            hue=pd.Series(y_resampled).value_counts().index, 
            palette='bright',
            legend=False)
plt.xticks(ticks=[0,1], labels=['Sehat', 'Stroke'])
plt.xlabel("Status")
plt.ylabel("Total")
plt.title("Distribusi Sesudah SMOTE")
plt.show()

print("\nTotal Data Sebelum SMOTE:")
print(X, y)

print("\nTotal Data Sesudah SMOTE:")
print(X_resampled, y_resampled)

# Simpan hasil sesudah SMOTE ke dalam file CSV
resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
resampled_df['stroke'] = y_resampled
resampled_df.to_csv("t-sne_oversampling.csv", index=False)
print("Hasil SMOTE telah disimpan dalam 't-sne_oversampling.csv'")

# Terapkan RandomUnderSampler
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)

# Cek distribusi sesudah undersampling
print("\nDistribusi sesudah Undersampling:")
print(pd.Series(y_resampled).value_counts())

# Plot distribusi sesudah undersampling
plt.figure(figsize=(6,4))
sns.barplot(x=pd.Series(y_resampled).value_counts().index,
            y=pd.Series(y_resampled).value_counts().values,
            hue=y.value_counts().index,
            palette='bright',
            legend=False)
plt.xticks(ticks=[0,1], labels=['Sehat', 'Stroke'])
plt.xlabel("Status")
plt.ylabel("Total")
plt.title("Distribusi Sesudah Undersampling")
plt.show()

print("\nTotal Data Sebelum UnderSampling:")
print(X, y)

print("\nTotal Data Sesudah UnderSampling:")
print(X_resampled, y_resampled)

# Simpan hasil sesudah Random UnderSampling ke dalam file CSV
under_df = pd.DataFrame(X_resampled, columns=X.columns)
under_df['stroke'] = y_resampled
under_df.to_csv("t-sne_undersampling.csv", index=False)
print("Hasil Random UnderSampling telah disimpan dalam 't-sne_undersampling.csv'")

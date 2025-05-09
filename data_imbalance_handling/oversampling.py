import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Gantilah dengan nama file dataset
df = pd.read_csv("stroke.csv")

# Bersihkan nama kolom dari spasi tersembunyi
df.columns = df.columns.str.strip()

# Cek apakah kolom 'stroke' ada
if 'stroke' not in df.columns:
    print("Error: Kolom 'stroke' tidak ditemukan dalam dataset.")
    exit()

# Menampilkan beberapa baris pertama
print(df.head())

# Diagram Batang stroke
stroke_counts = df['stroke'].value_counts()
stroke_counts.index = ['Sehat', 'Hipertensi']
print("\nJumlah Pasien Sehat dan Hipertensi:")
print(stroke_counts)

# Plot
plt.figure(figsize=(6,4))
sns.barplot(x=stroke_counts.index, 
            y=stroke_counts.values,
            hue=stroke_counts.index, 
            palette='bright',
            legend=False)

# Menambahkan label
plt.xlabel("Status")
plt.ylabel("Total")
plt.title("Total Pasien yang Sehat dan Hipertensi")
plt.show()

# Cek nilai yang hilang
missing = df.isna().sum()
print("\nJumlah Nilai yang Hilang dalam Dataset:")
print(missing)

# Isi nilai kosong pada kolom 'bmi' dengan median
df['bmi'] = df['bmi'].fillna(df['bmi'].median()).copy()

# Konversi kolom kategori menjadi numerik
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# --- Boxplot Sebelum Penanganan Outlier ---
plt.figure(figsize=(12, 4))
for i, col in enumerate(['age', 'avg_glucose_level', 'bmi']):
    plt.subplot(1, 3, i+1)
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot Sebelum - {col}")
plt.tight_layout()
plt.show()

# Deteksi dan Penanganan Outlier dengan Z-Score
numerical_columns = ['age', 'avg_glucose_level', 'bmi']
z_scores = df[numerical_columns].apply(zscore)
df = df[(z_scores < 3).all(axis=1)]

# --- Boxplot Setelah Penanganan Outlier dengan Z-Score ---
plt.figure(figsize=(12, 4))
for i, col in enumerate(numerical_columns):
    plt.subplot(1, 3, i+1)
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot Sesudah Z-Score - {col}")
plt.tight_layout()
plt.show()

# Menampilkan hasil dari Z-Score
print("\nZ-Score untuk Data Sebelum Penanganan Outlier:")
print(z_scores.describe())

# Menampilkan jumlah outlier (Z-Score > 3 atau < -3)
outlier_counts = (z_scores.abs() > 3).sum()
print("\nJumlah Outlier yang Terdeteksi dengan Z-Score:")
print(outlier_counts)

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
print("Distribusi Sebelum SMOTE:")
print(y.value_counts())
plt.figure(figsize=(6,4))
sns.barplot(x=y.value_counts().index, 
            y=y.value_counts().values,
            hue=y.value_counts().index, 
            palette='bright',
            legend=False)
plt.xticks(ticks=[0,1], labels=['Sehat', 'Hipertensi'])
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
plt.xticks(ticks=[0,1], labels=['Sehat', 'Hipertensi'])
plt.xlabel("Status")
plt.ylabel("Total")
plt.title("Distribusi Sesudah SMOTE")
plt.show()

print("\nTotal data setelah SMOTE:")
print(X_resampled, y_resampled)

# Simpan hasil setelah SMOTE ke dalam file CSV
resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
resampled_df['stroke'] = y_resampled
resampled_df.to_csv("stroke_oversampling.csv", index=False)
print("Hasil SMOTE telah disimpan dalam 'stroke_oversampling.csv'")

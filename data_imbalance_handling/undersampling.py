import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import zscore
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Gantilah dengan nama file dataset
df = pd.read_csv("/Volumes/DATA 1/UNAIR/SEMESTER 4/Machine Learning (Praktikum)/Imbalanced Data_187231090/stroke.csv")

# Bersihkan nama kolom dari spasi tersembunyi
df.columns = df.columns.str.strip()

# Cek apakah kolom 'hypertension' ada
if 'hypertension' not in df.columns:
    print("Error: Kolom 'hypertension' tidak ditemukan dalam dataset.")
    exit()

# Menampilkan beberapa baris pertama
print(df.head())

# Diagram Batang hypertension
hypertension_counts = df['hypertension'].value_counts()
hypertension_counts.index = ['Sehat', 'Hipertensi']
print("\nJumlah Pasien Sehat dan Hipertensi:")
print(hypertension_counts)

# Plot
plt.figure(figsize=(6,4))
sns.barplot(x=hypertension_counts.index, 
            y=hypertension_counts.values, 
            hue=hypertension_counts.index, 
            palette='bright',
            legend=False)

# Menambahkan label
plt.xlabel("Status")
plt.ylabel("Total")
plt.title("Total Pasien yang Hipertensi dan Sehat")
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

# Normalisasi fitur numerik
scaler = MinMaxScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Pilih fitur (X) dan target (y)
X = df.drop(columns=['id', 'hypertension'])  # Hapus ID & Target
y = df['hypertension']

# Terapkan RandomUnderSampler
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)

# Cek distribusi setelah undersampling
print("\nDistribusi Setelah Undersampling:")
print(pd.Series(y_resampled).value_counts())

# Plot distribusi setelah undersampling
plt.figure(figsize=(6,4))
sns.barplot(x=pd.Series(y_resampled).value_counts().index,
            y=pd.Series(y_resampled).value_counts().values,
            hue=y.value_counts().index,
            palette='bright',
            legend=False)
plt.xticks(ticks=[0,1], labels=['Sehat', 'Hipertensi'])
plt.xlabel("Status")
plt.ylabel("Total")
plt.title("Distribusi Setelah Undersampling")
plt.show()

print("\nTotal data sebelum UnderSampling:")
print(X, y)

print("\nTotal data setelah UnderSampling:")
print(X_resampled, y_resampled)

# Simpan hasil setelah Random UnderSampling ke dalam file CSV
under_df = pd.DataFrame(X_resampled, columns=X.columns)
under_df['stroke'] = y_resampled
under_df.to_csv("stroke_undersampling.csv", index=False)
print("Hasil Random UnderSampling telah disimpan dalam 'stroke_undersampling.csv'")

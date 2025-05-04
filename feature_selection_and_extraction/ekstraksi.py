import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Gantilah dengan nama file dataset
df = pd.read_csv("stroke.csv")

# Bersihkan nama kolom dari spasi tersembunyi
df.columns = df.columns.str.strip()

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

# Simpan hasil PCA ke CSV
df_pca.to_csv("pca_results.csv", index=False)
print("Hasil PCA telah disimpan dalam file pca_results.csv")

# Plot hasil PCA
plt.figure(figsize=(8, 6))
plt.scatter(df_pca["PC1"], df_pca["PC2"], c=df_pca["stroke"], cmap="rainbow", alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA: Stroke Dataset")
plt.colorbar(label="stroke")
plt.show()

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

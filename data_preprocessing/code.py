import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Membaca data dari CSV
file_path = "HousingData.csv"
df = pd.read_csv(file_path)

# 1. Cek missing value
print("\nCek missing value:")
print(df.head())
print("Missing value pada kolom:")
print(df.isnull().sum())

# 2. Menghapus missing value
print("\nMenghapus missing value:")
df.dropna(inplace=True)
print(df.head())
print("Missing values setelah dihapus:")
print(df.isnull().sum())

# 3. Mengisi missing value dengan mean
print("\nMengisi missing value dengan mean:")
df.fillna(df.mean(), inplace=True)
print(df.head())
print("Missing values setelah diisi:")
print(df.isnull().sum())

# 4. Mengisi missing value dengan median
print("\nMengisi missing value dengan median:")
df.fillna(df.median(), inplace=True)
print(df.head())
print("Missing values setelah diisi:")
print(df.isnull().sum())

# 5. Mengisi missing value dengan modus
print("\nMengisi missing value dengan modus:")
for col in df.select_dtypes(include=[np.number]).columns:
    mode_value = df[col].mode()[0]
    df[col] = df[col].fillna(mode_value)  # Directly assigning back
print(df.head())
print("Missing values setelah diisi:")
print(df.isnull().sum())

# 6. Deteksi outlier dengan Z-Score
print("\nDeteksi outlier dengan Z-Score:")
kolom_numerik = ['CRIM', 'RM', 'DIS']
z_scores = np.abs(stats.zscore(df[kolom_numerik]))
outliers_z = df[(z_scores > 3).any(axis=1)]
print("\nOutlier berdasarkan Z-score:\n", outliers_z)

# 7. Deteksi outlier dengan Boxplot (IQR)
print("\nDeteksi outlier dengan Boxplot (IQR):")
kolom_numerik = df.select_dtypes(include=[np.number]).columns  

# Fungsi untuk mendeteksi outlier menggunakan IQR
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

outliers_dict = {}
for col in kolom_numerik:
    outliers = detect_outliers_iqr(df, col)
    outliers_dict[col] = outliers
    print(f"Jumlah outlier di kolom {col}: {len(outliers)} data")

# Visualisasi boxplot untuk melihat outlier
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[kolom_numerik])
plt.xticks(rotation=45)
plt.title("Boxplot untuk Deteksi Outlier dengan IQR")
plt.show()

# 8. Menghapus outlier (Z-Score)
print("\nMenghapus outlier:")
z_scores = np.abs(stats.zscore(df[kolom_numerik]))
outliers_z = df[(z_scores > 3).any(axis=1)]
print("\nOutlier berdasarkan Z-score:\n", outliers_z)
df_cleaned = df[(z_scores <= 3).all(axis=1)]
print("\nDataset setelah menghapus outlier:\n", df_cleaned.head())

# 9. Mengganti outlier dengan mean atau median
print("\nMengganti outlier dengan mean atau median:")
kolom_numerik = df.select_dtypes(include=[np.number]).columns.tolist()
z_scores = df[kolom_numerik].apply(lambda x: stats.zscore(x, nan_policy='omit'))  # nan_policy='omit' agar mengabaikan NaN

outliers_z = df[(z_scores > 3).any(axis=1)]
print("\nOutlier berdasarkan Z-score:\n", outliers_z)

# Mengganti outlier dengan median di setiap kolom numerik
for col in kolom_numerik:
    median = df[col].median()
    df.loc[z_scores[col] > 3, col] = median

print("\nDataset setelah mengganti outlier dengan median:")
print(df.head())

# 10. Mengganti outlier dengan batas maksimum dan minimum IQR
print("\nMengganti outlier dengan batas maksimum dan minimum IQR:")
kolom_numerik = df.select_dtypes(include=[np.number]).columns

for col in kolom_numerik:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    batas_bawah = Q1 - 1.5 * IQR
    batas_atas = Q3 + 1.5 * IQR

    df[col] = np.where(df[col] < batas_bawah, batas_bawah, df[col]) 
    df[col] = np.where(df[col] > batas_atas, batas_atas, df[col])

print("\nDataset setelah mengganti outlier dengan batas maksimum dan minimum IQR:")
print(df.head())

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency, zscore

# Gantilah dengan nama file dataset
df = pd.read_csv("/Volumes/DATA 1/UNAIR/SEMESTER 4/Machine Learning (Praktikum)/Imbalanced Data_187231090/stroke.csv")

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

# Menampilkan hasil
print("\nDataset setelah konversi kategori:")
print(df.head())

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

# Uji Chi-Square untuk beberapa kolom
chi2_results = {}

for col in ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']:
    contingency_table = pd.crosstab(df[col], df['stroke'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    chi2_results[col] = {
        "chi2_value": chi2,
        "p_value": p,
        "dof": dof,
        "expected": expected
    }

# Menampilkan hasil uji Chi-Square
for col, result in chi2_results.items():
    print(f"\nUji Chi-Square untuk kolom '{col}':")
    print(f"Chi-Square Value: {result['chi2_value']}")
    print(f"P-Value: {result['p_value']}")
    print(f"Degrees of Freedom: {result['dof']}")
    print("Expected Frequencies:")
    print(result['expected'])

    # Menentukan apakah ada hubungan signifikan
    alpha = 0.05
    if result['p_value'] < alpha:
        print(f"Terdapat hubungan signifikan antara {col} dan Stroke")
    else:
        print(f"Tidak terdapat hubungan signifikan antara {col} dan Stroke.")

# Menggabungkan hasil uji Chi-Square dalam DataFrame
chi2_results_df = pd.DataFrame(columns=["Chi-Square Value", "P-Value", "Degrees of Freedom"])

for col, result in chi2_results.items():
    chi2_results_df.loc[col] = [result['chi2_value'], result['p_value'], result['dof']]

# Membuat DataFrame untuk hasil Z-Score dan Outlier
z_score_results_df = z_scores.describe()
outlier_counts_df = pd.DataFrame(outlier_counts, columns=["Outliers Detected"])

# Membuat DataFrame untuk nilai yang hilang
missing_df = missing.to_frame(name="Missing Values")

# Menggabungkan semua hasil ke dalam satu DataFrame
chi_square = pd.concat([missing_df, outlier_counts_df, z_score_results_df.T, chi2_results_df], axis=1)

# Menyimpan semua hasil dalam satu CSV
chi_square.to_csv("chi_square.csv")

print("Hasil telah disimpan dalam file chi_square.csv")

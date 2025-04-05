import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('/Volumes/DATA 1/UNAIR/SEMESTER 4/Machine Learning (Praktikum)/Tugas Transformasi_187231090/Cardiovascular_Disease_Dataset.csv')
df = pd.DataFrame(data)
print("\n Data Mentah:")
print(df, "\n")

# 1. Simple Feature Scaling pada seluruh data
# Normalisasi dengan Simple Feature Scaling (membagi dengan nilai maksimum untuk kolom numerik)
df_simple = df / df.max()
print("Data setelah Simple Feature Scaling:")
print(df_simple.head(), "\n")

df['serumcholestrol'] = df['serumcholestrol'] / df['serumcholestrol'].max()
df['maxheartrate'] = df['maxheartrate'] / df['maxheartrate'].max()
print("Data setelah Simple Feature Scaling (serumcholestrol dan maxheartrate):")
print(df[['serumcholestrol', 'maxheartrate']].head(), "\n")

# 2. Min-Max Normalization pada seluruh data
scaler_minmax = MinMaxScaler()
df_minmax = pd.DataFrame(scaler_minmax.fit_transform(df), columns=df.columns)
print("Data setelah Min-Max Normalization:")
print(df_minmax.head(), "\n")

scaler_minmax = MinMaxScaler()
df[['serumcholestrol', 'maxheartrate']] = scaler_minmax.fit_transform(df[['serumcholestrol', 'maxheartrate']])
print("Data setelah Min-Max Normalization (serumcholestrol dan maxheartrate):")
print(df[['serumcholestrol', 'maxheartrate']].head(), "\n")

# 3. Z-Score Standardization pada seluruh data
df_standardized = (df - df.mean()) / df.std()
print("Data setelah Z-Score Standardization:")
print(df_standardized.head(), "\n")

df['serumcholestrol'] = (df['serumcholestrol'] - df['serumcholestrol'].mean()) / df['serumcholestrol'].std()
df['maxheartrate'] = (df['maxheartrate'] - df['maxheartrate'].mean()) / df['maxheartrate'].std()
print("Data setelah Z-Score Standardization (serumcholestrol dan maxheartrate):")
print(df[['serumcholestrol', 'maxheartrate']].head())

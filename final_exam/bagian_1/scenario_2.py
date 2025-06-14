import numpy as np
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb

# 2 - Load Data
print("DATA AWAL".center(75, "=") + "\n")
dataframe = pd.read_excel('BlaBla.xlsx')
df = dataframe[['A', 'UMUR_TAHUN','B','C','D','E','F','G','H','I','J','K','L','M','N']]
df.columns = ['A', 'Umur','B','C','D','E','F','G','H','I','J','K','L','M','N']
print(df)

# Filter rows where UMUR_TAHUN is digit
df = df[df['Umur'].apply(lambda x: str(x).isdigit())]
df['Umur'] = df['Umur'].astype(int)

# 3.1 Cek Missing Values
print("CEK MISSING VALUE".center(75, "=") + "\n")
missing_values = df.isnull().sum()  # Mengecek nilai missing per kolom
print(missing_values)

# 3.2 Imputasi Missing Values (Jika ada) - Menggunakan rata-rata untuk kolom numerik
df = df.fillna(df.mean())  # Mengisi missing value dengan rata-rata kolom

# 3.3 Deteksi Outlier Menggunakan Z-Score
print("\n" + "DETEKSI OUTLIER DENGAN Z-SCORE".center(75, "="))

# Hitung Z-Score untuk setiap kolom numerik
z_scores = np.abs(zscore(df.iloc[:, 0:13]))  # Menghitung Z-Score untuk fitur numerik saja

# Tentukan ambang batas Z-Score (misalnya 3)
threshold = 3

# Menentukan outlier berdasarkan Z-Score lebih besar dari threshold
outliers = (z_scores > threshold)

# Menampilkan data yang dianggap outlier
outlier_indices = np.where(outliers)[0]
outlier_columns = np.where(outliers)[1]

# Data outlier yang terdeteksi
outlier_data = df.iloc[outlier_indices]

print(f"\nJumlah outlier yang terdeteksi: {len(outlier_data)}")
print("\nData Outlier Terdeteksi:")
print(outlier_data)

# Menghapus outlier yang terdeteksi
df_cleaned = df.drop(index=outlier_indices, axis=0, errors='ignore')

# Reset indeks untuk memperbaiki urutan indeks setelah penghapusan
df_cleaned = df_cleaned.reset_index(drop=True)

# Menampilkan dataset setelah outlier dihapus
print(f"\nJumlah data setelah penghapusan outlier: {len(df_cleaned)}")
print(df_cleaned)

# 4 - Encoding Umur
def encode_age(age):
    if age <= 20: return 1
    elif age <= 30: return 2
    elif age <= 40: return 3
    elif age <= 50: return 4
    else: return 5

df['Umur_Kategori'] = df['Umur'].apply(encode_age)
df.drop(columns=['Umur'], inplace=True)

# 5 - Split Feature & Label
print("\n" + "GROUPING VARIABEL".center(75, "="))
X = df.iloc[:, 0:13].values  # Features (X)
y = df.iloc[:, 13].values    # Labels (y)
print("\nData Variabel (X):\n", X)
print("\nData Label (y):\n", y)

# 5.1 - Visualisasi Sebelum Undersampling (Distribusi Kelas)
plt.figure(figsize=(8, 6))
sns.countplot(x='N', data=df, palette='Set2')
plt.title('Distribusi Kelas Sebelum Undersampling')
plt.xlabel('Label (Kelas)')
plt.ylabel('Jumlah')
plt.show()

# 5.2 - Undersampling Kelas Mayoritas
# Gabungkan fitur dan label untuk melakukan undersampling secara bersamaan
df_combined = df.copy()

# Identifikasi jumlah kelas mayoritas dan minoritas
majority_class = df_combined[df_combined['N'] == 0]  # Misalkan kelas mayoritas adalah '0'
minority_class = df_combined[df_combined['N'] == 1]  # Misalkan kelas minoritas adalah '1'

# Lakukan undersampling pada kelas mayoritas
majority_class_undersampled = resample(majority_class, 
                                       replace=False,    # Tidak mengizinkan sampling dengan penggantian
                                       n_samples=len(minority_class),  # Sesuaikan jumlahnya dengan kelas minoritas
                                       random_state=0)

# Gabungkan kembali kelas mayoritas yang telah diundersample dengan kelas minoritas
df_balanced = pd.concat([majority_class_undersampled, minority_class])

# Pisahkan kembali menjadi fitur (X) dan label (y)
X_balanced = df_balanced.iloc[:, 0:13].values  # Features
y_balanced = df_balanced.iloc[:, 13].values    # Labels

# 5.3 - Visualisasi Setelah Undersampling (Distribusi Kelas)
plt.figure(figsize=(8, 6))
sns.countplot(x='N', data=df_balanced, palette='Set2')
plt.title('Distribusi Kelas Setelah Undersampling')
plt.xlabel('Label (Kelas)')
plt.ylabel('Jumlah')
plt.show()

# 5.4 - Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_balanced)

# 6 - Train-Test Split
print("\n" + "SPLITTING DATA 80-20".center(75, "="))
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_balanced, test_size=0.2, random_state=0)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# 7 - Model Decision Tree
print("\n" + "MODEL DECISION TREE".center(75, "="))
decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree.fit(X_train, y_train)
y_pred_dt = decision_tree.predict(X_test)
print("\nPrediksi Decision Tree:\n", y_pred_dt)

# 8 - Model Random Forest
print("\n" + "MODEL RANDOM FOREST".center(75, "="))
random_forest = RandomForestClassifier(random_state=0)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
print("\nPrediksi Random Forest:\n", y_pred_rf)

# 9 - Model SVM
print("\n" + "MODEL SVM".center(75, "="))
svm_model = SVC(probability=True, random_state=0)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
print("\nPrediksi SVM:\n", y_pred_svm)

# 10 - Model XGBoost
print("\n" + "MODEL XGBoost".center(75, "="))
xgb_model = xgb.XGBClassifier(random_state=0)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print("\nPrediksi XGBoost:\n", y_pred_xgb)

# 11 - Model LightGBM
print("\n" + "MODEL LightGBM".center(75, "="))
lgb_model = lgb.LGBMClassifier(random_state=0)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
print("\nPrediksi LightGBM:\n", y_pred_lgb)

# 12 - Fungsi Evaluasi Model
def evaluasi_model(nama, y_true, y_pred, y_prob=None):
    print("\n" + f"{nama.center(75, '=')}")
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(cls) for cls in np.unique(y_true)])
    disp.plot(cmap=plt.cm.cool)
    plt.gcf().set_facecolor('black')
    plt.gca().set_facecolor('black')
    plt.xlabel('Predicted label', color='white')
    plt.ylabel('True label', color='white')
    plt.title(f"{nama} - Confusion Matrix", color='white')
    disp.ax_.tick_params(axis='x', colors='white')
    disp.ax_.tick_params(axis='y', colors='white')
    for text in disp.ax_.texts:
        text.set_color('black')
    colorbar = disp.ax_.images[-1].colorbar
    colorbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(colorbar.ax.yaxis.get_ticklabels(), color='white')    
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=3))
    print("Akurasi :", round(accuracy_score(y_true, y_pred) * 100, 3), "%")
    print("Presisi :", round(precision_score(y_true, y_pred, zero_division=0), 3))
    print("Recall  :", round(recall_score(y_true, y_pred, zero_division=0), 3))
    print("F1 Score:", round(f1_score(y_true, y_pred, zero_division=0), 3))
    if y_prob is not None:
        print("AUC-ROC :", round(roc_auc_score(y_true, y_prob), 3))

# 13 - Evaluasi Semua Model
# Probabilitas untuk AUC-ROC
y_prob_dt = decision_tree.predict_proba(X_test)[:, 1]
y_prob_rf = random_forest.predict_proba(X_test)[:, 1]
y_prob_svm = svm_model.predict_proba(X_test)[:, 1]
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]

# Evaluasi untuk masing-masing model
evaluasi_model("DECISION TREE", y_test, y_pred_dt, y_prob_dt)
evaluasi_model("RANDOM FOREST", y_test, y_pred_rf, y_prob_rf)
evaluasi_model("SVM", y_test, y_pred_svm, y_prob_svm)
evaluasi_model("XGBoost", y_test, y_pred_xgb, y_prob_xgb)
evaluasi_model("LightGBM", y_test, y_pred_lgb, y_prob_lgb)

# 14 - Memilih Model Terbaik Berdasarkan AUC-ROC
roc_auc_dt = roc_auc_score(y_test, y_prob_dt)
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)
roc_auc_svm = roc_auc_score(y_test, y_prob_svm)
roc_auc_xgb = roc_auc_score(y_test, y_prob_xgb)
roc_auc_lgb = roc_auc_score(y_test, y_prob_lgb)

roc_auc_scores = {
    "Decision Tree": roc_auc_dt,
    "Random Forest": roc_auc_rf,
    "SVM": roc_auc_svm,
    "XGBoost": roc_auc_xgb,
    "LightGBM": roc_auc_lgb
}

best_model_name = max(roc_auc_scores, key=roc_auc_scores.get)
print(f"\nModel terbaik berdasarkan AUC-ROC: {best_model_name}\n")

# 15 - ROC Curve
plt.figure(figsize=(8, 6))

# Plot ROC untuk semua model
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
fpr_lgb, tpr_lgb, _ = roc_curve(y_test, y_prob_lgb)

plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {roc_auc_dt:.3f})', color='skyblue')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.3f})', color='violet')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {roc_auc_svm:.3f})', color='green')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.3f})', color='orange')
plt.plot(fpr_lgb, tpr_lgb, label=f'LightGBM (AUC = {roc_auc_lgb:.3f})', color='red')

# Plot diagonal line
plt.plot([0, 1], [0, 1], color='white', linestyle='--')

# Formatting
plt.gca().set_facecolor('black')
plt.gcf().set_facecolor('black')
plt.xlabel('False Positive Rate', color='white')
plt.ylabel('True Positive Rate', color='white')
plt.title('ROC Curve for All Models', color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.grid(color='white', linestyle='--', linewidth=0.5)

legend = plt.legend(loc='lower right', frameon=False)
for text in legend.get_texts():
    text.set_color('white')

# Show ROC curve
plt.show()

# 16 - Menyusun Tabel Evaluasi untuk Model
def hitung_metrix(y_true, y_pred, y_prob):
    akurasi = accuracy_score(y_true, y_pred)
    presisi = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_true, y_prob)
    return akurasi, presisi, recall, f1, auc_roc

# Data untuk tabel evaluasi
eval_data = []

# Perhitungan untuk Decision Tree
y_pred_dt = decision_tree.predict(X_test)
y_prob_dt = decision_tree.predict_proba(X_test)[:, 1]
metrics_dt = hitung_metrix(y_test, y_pred_dt, y_prob_dt)
eval_data.append(["Decision Tree"] + list(metrics_dt))

# Perhitungan untuk Random Forest
y_pred_rf = random_forest.predict(X_test)
y_prob_rf = random_forest.predict_proba(X_test)[:, 1]
metrics_rf = hitung_metrix(y_test, y_pred_rf, y_prob_rf)
eval_data.append(["Random Forest"] + list(metrics_rf))

# Perhitungan untuk SVM
y_pred_svm = svm_model.predict(X_test)
y_prob_svm = svm_model.predict_proba(X_test)[:, 1]
metrics_svm = hitung_metrix(y_test, y_pred_svm, y_prob_svm)
eval_data.append(["SVM"] + list(metrics_svm))

# Perhitungan untuk XGBoost
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
metrics_xgb = hitung_metrix(y_test, y_pred_xgb, y_prob_xgb)
eval_data.append(["XGBoost"] + list(metrics_xgb))

# Perhitungan untuk LightGBM
y_pred_lgb = lgb_model.predict(X_test)
y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
metrics_lgb = hitung_metrix(y_test, y_pred_lgb, y_prob_lgb)
eval_data.append(["LightGBM"] + list(metrics_lgb))

# Tabel Evaluasi
eval_df = pd.DataFrame(eval_data, columns=["Model", "Akurasi", "Presisi", "Recall", "F1-Score", "AUC-ROC"])

# Tampilkan Tabel Evaluasi tanpa nomor index
print("\nTabel Evaluasi Model:")
print(eval_df.to_string(index=False))

# Model Terbaik Berdasarkan AUC-ROC
best_model = eval_df.loc[eval_df["AUC-ROC"].idxmax()]
print(f"\nModel terbaik berdasarkan AUC-ROC: {best_model['Model']} dengan AUC-ROC = {best_model['AUC-ROC']:.3f}") 

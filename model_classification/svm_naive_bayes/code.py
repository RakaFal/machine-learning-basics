import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, auc, 
                             ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# 2 - Load Data
print("DATA AWAL".center(75, "="))
dataframe = pd.read_excel('/Volumes/DATA 1/UNAIR/SEMESTER 4/Machine Learning (Praktikum)/DT_RF_187231090/BlaBla.xlsx')
data = dataframe[['A','B','C','D','E','F','G','H','I','J','K','L','M','N']]
print(data)
print("="*75)

# 3 - Split Feature & Label
print("GROUPING VARIABEL".center(75, "="))
X = data.iloc[:, 0:13].values
y = data.iloc[:, 13].values
print("Data Variabel (X):\n", X)
print("Data Label (y):\n", y)
print("="*75)

# 4 - Train-Test Split
print("SPLITTING DATA 80-20".center(75, "="))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print("="*75)

# 5 - Model - SVM
print("MODEL SVM".center(75, "="))
svm_model = SVC(probability=True, random_state=0)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
print("Prediksi SVM:\n", y_pred_svm)
print("="*75)

# 6 - Model - Naive Bayes
print("MODEL NAIVE BAYES".center(75, "="))
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
print("Prediksi Naive Bayes:\n", y_pred_nb)
print("="*75)

# 7 - Model - KNN
print("MODEL KNN".center(75, "="))
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
print("Prediksi KNN:\n", y_pred_knn)
print("="*75)

# 8 - Model - Logistic Regression
print("MODEL LOGISTIC REGRESSION".center(75, "="))
lr_model = LogisticRegression(max_iter=1000, random_state=0)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
print("Prediksi Logistic Regression:\n", y_pred_lr)
print("="*75)

# 9 - Model - Gradient Boosting
print("MODEL GRADIENT BOOSTING".center(75, "="))
gb_model = GradientBoostingClassifier(random_state=0)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
print("Prediksi Gradient Boosting:\n", y_pred_gb)
print("="*75)

# 10 - Fungsi Evaluasi Model
def evaluasi_model(nama, y_true, y_pred, y_prob=None):
    print(f"{nama.center(75, '=')}")
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
    print("="*75)

# 11 - Evaluasi Semua Model
y_prob_svm = svm_model.predict_proba(X_test)[:, 1]
y_prob_nb = nb_model.predict_proba(X_test)[:, 1]
y_prob_knn = knn_model.predict_proba(X_test)[:, 1]
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]
y_prob_gb = gb_model.predict_proba(X_test)[:, 1]

print("\nEvaluasi Model SVM:")
evaluasi_model("SVM", y_test, y_pred_svm, y_prob_svm)
print("\nEvaluasi Model Naive Bayes:")
evaluasi_model("NAIVE BAYES", y_test, y_pred_nb, y_prob_nb)
print("\nEvaluasi Model KNN:")
evaluasi_model("KNN", y_test, y_pred_knn, y_prob_knn)
print("\nEvaluasi Model Logistic Regression:")
evaluasi_model("LOGISTIC REGRESSION", y_test, y_pred_lr, y_prob_lr)
print("\nEvaluasi Model Gradient Boosting:")
evaluasi_model("GRADIENT BOOSTING", y_test, y_pred_gb, y_prob_gb)

# 12 - Memilih Model Terbaik Berdasarkan AUC-ROC
roc_aucs = {
    "SVM": roc_auc_score(y_test, y_prob_svm),
    "Naive Bayes": roc_auc_score(y_test, y_prob_nb),
    "KNN": roc_auc_score(y_test, y_prob_knn),
    "Logistic Regression": roc_auc_score(y_test, y_prob_lr),
    "Gradient Boosting": roc_auc_score(y_test, y_prob_gb)
}
best_model_name = max(roc_aucs, key=roc_aucs.get)
print(f"Model terbaik: {best_model_name}\n")
model_dict = {
    "SVM": svm_model,
    "Naive Bayes": nb_model,
    "KNN": knn_model,
    "Logistic Regression": lr_model,
    "Gradient Boosting": gb_model
}
best_model = model_dict[best_model_name]

# 13 - ROC Curve Semua Model
plt.figure(figsize=(10, 8))
for name, prob in zip(roc_aucs.keys(), [y_prob_svm, y_prob_nb, y_prob_knn, y_prob_lr, y_prob_gb]):
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, prob):.3f})")
plt.plot([0, 1], [0, 1], 'k--', color='white')
plt.gca().set_facecolor('black')
plt.gcf().set_facecolor('black')
plt.xlabel('False Positive Rate', color='white')
plt.ylabel('True Positive Rate', color='white')
plt.title('ROC Curve', color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.grid(color='white', linestyle='--', linewidth=0.5)
legend = plt.legend(loc='lower right', frameon=False)
for text in legend.get_texts():
    text.set_color('white')
plt.show()

# 14 - Input Example dari User
def get_input():
    """Function to handle user input"""
    try:
        A = int(input("Umur Pasien = "))
        if not (0 <= A <= 120):
            print("Umur tidak valid! Harap masukkan umur antara 0-120.")
            return None

        print("Isi Jenis Kelamin dengan 0 jika Perempuan dan 1 jika Laki-Laki")
        B = int(input("Jenis Kelamin Pasien = "))
        print("Isi Y jika mengalami dan N jika tidak")
        C = input("Apakah pasien mengalami C? = ").upper()
        D = input("Apakah pasien mengalami D? = ").upper()
        E = input("Apakah pasien mengalami E? = ").upper()
        F = input("Apakah pasien mengalami F? = ").upper()
        G = input("Apakah pasien mengalami G? = ").upper()
        H = input("Apakah pasien mengalami H? = ").upper()
        I = input("Apakah pasien mengalami I? = ").upper()
        J = input("Apakah pasien mengalami J? = ").upper()
        K = input("Apakah pasien mengalami K? = ").upper()
        L = input("Apakah pasien mengalami L? = ").upper()
        M = input("Apakah M? = ").upper()

        A_k = 0
        if A < 21: A_k = 1
        elif 21 <= A < 31: A_k = 2
        elif 31 <= A < 41: A_k = 3
        elif 41 <= A < 51: A_k = 4
        elif A >= 51: A_k = 5
        print("Kode umur pasien adalah", A_k)

        B_k = 1 if B == 1 else 0

        binary_conversion = {'Y': 1, 'N': 0}
        inputs = [A_k, B_k] + [binary_conversion.get(i, 0) for i in [C, D, E, F, G, H, I, J, K, L, M]]
        print("Input pasien:", inputs)
        return np.array(inputs).reshape(1, -1)
    except ValueError:
        print("Input tidak valid. Harap masukkan nilai yang benar.")
        return None

print("CONTOH INPUT".center(75, "="))
input_pasien = get_input()
if input_pasien is not None:
    predtest = best_model.predict(input_pasien)
    print("Pasien Positive" if predtest == 1 else "Pasien Negative")
print("="*75)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, auc, 
                             ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

# 2 - Load Data
print("DATA AWAL".center(75, "="))
dataframe = pd.read_excel('BlaBla.xlsx')
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
print("SPLITTING DATA 20-80".center(75, "="))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print("="*75)

# 5 - Decision Tree Model
print("MODEL DECISION TREE".center(75, "="))
decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree.fit(X_train, y_train)
y_pred_dt = decision_tree.predict(X_test)
print("Prediksi Decision Tree:\n", y_pred_dt)
print("="*75)

# 6 - Visualisasi Decision Tree
print("VISUALISASI DECISION TREE".center(75, "="))
plt.figure(figsize=(25, 10))
plot_tree(
    decision_tree,
    feature_names=data.columns[0:13],
    class_names=[str(cls) for cls in np.unique(y)],
    filled=True,
    rounded=True,
    fontsize=4
)
plt.gca().set_facecolor('gray')
plt.gcf().set_facecolor('gray')
plt.title("Decision Tree Visualization", color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.show()
print("="*75)

# 7 - Random Forest Model
print("MODEL RANDOM FOREST".center(75, "="))
random_forest = RandomForestClassifier(random_state=0)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
print("Prediksi Random Forest:\n", y_pred_rf)
print("="*75)

# 8 - Visualisasi Salah Satu Pohon di Random Forest
print("VISUALISASI SALAH SATU POHON DI RANDOM FOREST".center(75, "="))
plt.figure(figsize=(25, 10))
plot_tree(
    random_forest.estimators_[0],
    feature_names=data.columns[0:13],
    class_names=[str(cls) for cls in np.unique(y)],
    filled=True,
    rounded=True,
    fontsize=4
)
plt.gca().set_facecolor('gray')
plt.gcf().set_facecolor('gray')
plt.title("Salah Satu Pohon di Random Forest", color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.show()
print("="*75)

# 9 - Fungsi Evaluasi Model
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

# 10 - Evaluasi Decision Tree & Random Forest
y_prob_dt = decision_tree.predict_proba(X_test)[:, 1]
y_prob_rf = random_forest.predict_proba(X_test)[:, 1]
print("\nEvaluasi Model Decision Tree:")
evaluasi_model("DECISION TREE", y_test, y_pred_dt, y_prob_dt)
print("\nEvaluasi Model Random Forest:")
evaluasi_model("RANDOM FOREST", y_test, y_pred_rf, y_prob_rf)

# 11 - Memilih Model Terbaik Berdasarkan AUC-ROC
# Hitung AUC-ROC untuk kedua model
roc_auc_dt = roc_auc_score(y_test, y_prob_dt)
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)
if roc_auc_dt > roc_auc_rf:
    print("Model terbaik: Decision Tree\n")
    best_model = decision_tree
else:
    print("Model terbaik: Random Forest\n")
    best_model = random_forest

# 11 - ROC Curve
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_dt = auc(fpr_dt, tpr_dt)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.figure(figsize=(8, 6))
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {roc_auc_dt:.3f})', color='skyblue')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.3f})', color='violet')
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

# 12 - Input Example from User (After Models are Trained)
def get_input():
    """Function to handle user input"""
    try:
        A = int(input("Umur Pasien = "))
        if not (0 <= A <= 120):  # Age should be between 0 and 120
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

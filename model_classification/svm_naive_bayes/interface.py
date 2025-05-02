import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import streamlit as st

# Streamlit app layout
st.title("Machine Learning with Naïve Bayes, SVM, KNN, Logistic Regression, and Gradient Boosting")
st.write("### Data and Model Evaluation")

# 2 - Load Data
st.write("#### Data Overview")
dataframe = pd.read_excel('/Volumes/DATA 1/UNAIR/SEMESTER 4/Machine Learning (Praktikum)/Tugas klasifikasi_187231090/BlaBla.xlsx')
data = dataframe[['A','B','C','D','E','F','G','H','I','J','K','L','M','N']]
st.write(data)
st.write("="*88)

# 3 - Split Feature & Label
st.write("#### Grouping Variables")
X = data.iloc[:, 0:13].values
y = data.iloc[:, 13].values
st.write("Data Variabel (X):", X)
st.write("Data Label (y):", y)
st.write("="*88)

# 4 - Train-Test Split
st.write("#### Train-Test Split")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
st.write(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
st.write("="*88)

# 5 - SVM Model
st.write("#### SVM Model")
svm_model = SVC(probability=True, random_state=0)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
st.write("Prediksi SVM:", y_pred_svm)
st.write("="*88)

# 6 - Naive Bayes Model
st.write("#### Naive Bayes Model")
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
st.write("Prediksi Naive Bayes:", y_pred_nb)
st.write("="*88)

# 7 - KNN Model
st.write("#### KNN Model")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
st.write("Prediksi KNN:", y_pred_knn)
st.write("="*88)

# 8 - Logistic Regression Model
st.write("#### Logistic Regression Model")
lr_model = LogisticRegression(max_iter=1000, random_state=0)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
st.write("Prediksi Logistic Regression:", y_pred_lr)
st.write("="*88)

# 9 - Gradient Boosting Model
st.write("#### Gradient Boosting Model")
gb_model = GradientBoostingClassifier(random_state=0)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
st.write("Prediksi Gradient Boosting:", y_pred_gb)
st.write("="*88)

# 10 - Fungsi Evaluasi Model
def evaluasi_model(nama, y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred)
    st.write("Confusion Matrix:")
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
    st.pyplot(plt)

    st.write("Classification Report:")
    st.write("Akurasi :", round(accuracy_score(y_true, y_pred) * 100, 3), "%")
    st.write("Presisi :", round(precision_score(y_true, y_pred, zero_division=0), 3))
    st.write("Recall  :", round(recall_score(y_true, y_pred, zero_division=0), 3))
    st.write("F1 Score:", round(f1_score(y_true, y_pred, zero_division=0), 3))
    if y_prob is not None:
        st.write("AUC-ROC :", round(roc_auc_score(y_true, y_prob), 3))
    st.write("="*88)

# 11 - Evaluasi Semua Model
y_prob_svm = svm_model.predict_proba(X_test)[:, 1]
y_prob_nb = nb_model.predict_proba(X_test)[:, 1]
y_prob_knn = knn_model.predict_proba(X_test)[:, 1]
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]
y_prob_gb = gb_model.predict_proba(X_test)[:, 1]

st.write("Evaluasi Model SVM:")
evaluasi_model("SVM", y_test, y_pred_svm, y_prob_svm)
st.write("Evaluasi Model Naive Bayes:")
evaluasi_model("NAIVE BAYES", y_test, y_pred_nb, y_prob_nb)
st.write("Evaluasi Model KNN:")
evaluasi_model("KNN", y_test, y_pred_knn, y_prob_knn)
st.write("Evaluasi Model Logistic Regression:")
evaluasi_model("LOGISTIC REGRESSION", y_test, y_pred_lr, y_prob_lr)
st.write("Evaluasi Model Gradient Boosting:")
evaluasi_model("GRADIENT BOOSTING", y_test, y_pred_gb, y_prob_gb)

# 12 - ROC Curve Semua Model
roc_aucs = {
    "SVM": roc_auc_score(y_test, y_prob_svm),
    "Naive Bayes": roc_auc_score(y_test, y_prob_nb),
    "KNN": roc_auc_score(y_test, y_prob_knn),
    "Logistic Regression": roc_auc_score(y_test, y_prob_lr),
    "Gradient Boosting": roc_auc_score(y_test, y_prob_gb)
}

st.write("ROC Curve:")
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
st.pyplot(plt)
st.write("="*88)

# 13 - Memilih Model Terbaik Berdasarkan AUC-ROC
best_model_name = max(roc_aucs, key=roc_aucs.get)
st.write(f"Model terbaik: {best_model_name}")
model_dict = {
    "SVM": svm_model,
    "Naive Bayes": nb_model,
    "KNN": knn_model,
    "Logistic Regression": lr_model,
    "Gradient Boosting": gb_model
}
best_model = model_dict[best_model_name]
st.write("="*88)

# 14 - Input Example from User
st.write("#### Input Example")
def get_input():
    """Function to handle user input"""
    try:
        A = st.slider("Umur Pasien", 0, 120, 30)
        B = st.radio("Jenis Kelamin Pasien",options=[0, 1], format_func=lambda x: "Perempuan" if x == 0 else "Laki-Laki", horizontal=True)
        C = st.radio("Apakah pasien mengalami C?", ["Y", "N"], horizontal=True)
        D = st.radio("Apakah pasien mengalami D?", ["Y", "N"], horizontal=True)
        E = st.radio("Apakah pasien mengalami E?", ["Y", "N"], horizontal=True)
        F = st.radio("Apakah pasien mengalami F?", ["Y", "N"], horizontal=True)
        G = st.radio("Apakah pasien mengalami G?", ["Y", "N"], horizontal=True)
        H = st.radio("Apakah pasien mengalami H?", ["Y", "N"], horizontal=True)
        I = st.radio("Apakah pasien mengalami I?", ["Y", "N"], horizontal=True)
        J = st.radio("Apakah pasien mengalami J?", ["Y", "N"], horizontal=True)
        K = st.radio("Apakah pasien mengalami K?", ["Y", "N"], horizontal=True)
        L = st.radio("Apakah pasien mengalami L?", ["Y", "N"], horizontal=True)
        M = st.radio("Apakah pasien mengalami M?", ["Y", "N"], horizontal=True)

        A_k = 0
        if A < 21: 
            A_k = 1
        elif 21 <= A < 31: 
            A_k = 2
        elif 31 <= A < 41: 
            A_k = 3
        elif 41 <= A < 51: 
            A_k = 4
        elif A >= 51: 
            A_k = 5

        binary_conversion = {'Y': 1, 'N': 0}
        inputs = [A_k, B] + [binary_conversion.get(i, 0) for i in [C, D, E, F, G, H, I, J, K, L, M]]
        st.write("Input pasien:", inputs)
        return np.array(inputs).reshape(1, -1)
    except ValueError:
        st.error("Input tidak valid. Harap masukkan nilai yang benar.")
        return None

input_pasien = get_input()
if input_pasien is not None:
    predtest = best_model.predict(input_pasien)
    st.write("Pasien Positive" if predtest == 1 else "Pasien Negative")
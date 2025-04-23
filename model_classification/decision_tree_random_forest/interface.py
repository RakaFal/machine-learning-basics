import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Function to evaluate models
def evaluasi_model(nama, y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred)
    st.write("Confusion Matrix:")
    st.write(cm)
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

    st.write("\nClassification Report:")
    st.write(classification_report(y_true, y_pred, digits=3))
    st.write(f"Akurasi : {round(accuracy_score(y_true, y_pred) * 100, 3)} %")
    st.write(f"Presisi : {round(precision_score(y_true, y_pred, zero_division=0), 3)}")
    st.write(f"Recall  : {round(recall_score(y_true, y_pred, zero_division=0), 3)}")
    st.write(f"F1 Score: {round(f1_score(y_true, y_pred, zero_division=0), 3)}")
    if y_prob is not None:
        st.write(f"AUC-ROC : {round(roc_auc_score(y_true, y_prob), 3)}")
    st.write("="*88)

# Streamlit app layout
st.title("Machine Learning with Decision Tree and Random Forest")
st.write("### Example Data and Model Evaluation")

# 2 - Load Data
st.write("#### Data Overview")
dataframe = pd.read_excel('/Volumes/DATA 1/UNAIR/SEMESTER 4/Machine Learning (Praktikum)/DT_RF_187231090/BlaBla.xlsx')
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

# 5 - Decision Tree Model
st.write("#### Decision Tree Model")
decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree.fit(X_train, y_train)
y_pred_dt = decision_tree.predict(X_test)
st.write("Prediksi Decision Tree:", y_pred_dt)
st.write("="*88)

# 6 - Visualisasi Decision Tree
st.write("#### Decision Tree Visualization")
fig, ax = plt.subplots(figsize=(25, 10))
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
st.pyplot(fig)
st.write("="*88)

# 7 - Random Forest Model
st.write("#### Random Forest Model")
random_forest = RandomForestClassifier(random_state=0)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
st.write("Prediksi Random Forest:", y_pred_rf)
st.write("="*88)

# 8 - Visualisasi Salah Satu Pohon di Random Forest
st.write("#### Salah Satu Pohon di Random Forest")
fig, ax = plt.subplots(figsize=(25, 10))
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
st.pyplot(fig)
st.write("="*88)

# 9 - Evaluasi Model Decision Tree & Random Forest
y_prob_dt = decision_tree.predict_proba(X_test)[:, 1]
y_prob_rf = random_forest.predict_proba(X_test)[:, 1]
st.write("#### Evaluasi Model Decision Tree:")
evaluasi_model("DECISION TREE", y_test, y_pred_dt, y_prob_dt)
st.write("#### Evaluasi Model Random Forest:")
evaluasi_model("RANDOM FOREST", y_test, y_pred_rf, y_prob_rf)

# 10 - Memilih Model Terbaik Berdasarkan AUC-ROC
roc_auc_dt = roc_auc_score(y_test, y_prob_dt)
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)
best_model = decision_tree if roc_auc_dt > roc_auc_rf else random_forest
st.write(f"Model terbaik: {'Decision Tree' if roc_auc_dt > roc_auc_rf else 'Random Forest'}")
st.write("="*88)

# 11 - ROC Curve
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_dt = auc(fpr_dt, tpr_dt)
roc_auc_rf = auc(fpr_rf, tpr_rf)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {roc_auc_dt:.3f})', color='skyblue')
ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.3f})', color='violet')
ax.plot([0, 1], [0, 1], 'k--', color='white')
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
st.pyplot(fig)
st.write("="*88)

# 12 - Input Example from User
st.write("#### Input Example")
def get_input():
    """Function to handle user input"""
    try:
        A = st.slider("Umur Pasien", 0, 120, 30)
        B = st.selectbox("Jenis Kelamin Pasien", [0, 1])  # 0: Perempuan, 1: Laki-Laki
        C = st.radio("Apakah pasien mengalami C?", ["Y", "N"])
        D = st.radio("Apakah pasien mengalami D?", ["Y", "N"])
        E = st.radio("Apakah pasien mengalami E?", ["Y", "N"])
        F = st.radio("Apakah pasien mengalami F?", ["Y", "N"])
        G = st.radio("Apakah pasien mengalami G?", ["Y", "N"])
        H = st.radio("Apakah pasien mengalami H?", ["Y", "N"])
        I = st.radio("Apakah pasien mengalami I?", ["Y", "N"])
        J = st.radio("Apakah pasien mengalami J?", ["Y", "N"])
        K = st.radio("Apakah pasien mengalami K?", ["Y", "N"])
        L = st.radio("Apakah pasien mengalami L?", ["Y", "N"])
        M = st.radio("Apakah pasien mengalami M?", ["Y", "N"])

        A_k = 0
        if A < 21: A_k = 1
        elif 21 <= A < 31: A_k = 2
        elif 31 <= A < 41: A_k = 3
        elif 41 <= A < 51: A_k = 4
        elif A >= 51: A_k = 5

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
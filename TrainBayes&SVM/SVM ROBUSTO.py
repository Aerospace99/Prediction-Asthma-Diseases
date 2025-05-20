import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import ADASYN
from collections import Counter as ctr
import seaborn as sns
import matplotlib.pyplot as plt

# --- Función de preprocesamiento ---
def preprocess(df):
    df = df.copy()
    df.drop(['PatientID', 'DoctorInCharge'], axis=1, errors='ignore', inplace=True)

    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 19, 60, 100], labels=['Child', 'Teen', 'Adult', 'Senior'], right=False)
    df['BMICategory'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'], right=False)

    df['LifestyleScore'] = df[['PhysicalActivity', 'DietQuality', 'SleepQuality']].mean(axis=1)
    df['AllergyScore'] = df[['PetAllergy', 'HistoryOfAllergies', 'Eczema', 'HayFever']].sum(axis=1)
    df['ExposureScore'] = df[['PollutionExposure', 'PollenExposure', 'DustExposure']].mean(axis=1)
    df['SymptomSeverityScore'] = df[['Wheezing', 'ShortnessOfBreath', 'ChestTightness', 'Coughing', 'NighttimeSymptoms', 'ExerciseInduced']].sum(axis=1)
    df['LungFunctionRatio'] = df['LungFunctionFEV1'] / df['LungFunctionFVC']

    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    return df

# --- Cargar y procesar datos de entrenamiento ---
df_train = pd.read_csv("asthma_disease_data.csv")
df_train = preprocess(df_train)

X = df_train.drop('Diagnosis', axis=1)
y = df_train['Diagnosis']

# Selección de características
selector = SelectKBest(score_func=f_classif, k=min(15, X.shape[1]))
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print("Características seleccionadas:", list(selected_features))

# División de datos
X_train, X_val, y_train, y_val = train_test_split(X[selected_features], y, test_size=0.3, random_state=40)

# --- Aplicar ADASYN para balancear las clases ---
print("Distribución antes de ADASYN:", ctr(y_train))
adasyn = ADASYN(random_state=40)
X_train, y_train = adasyn.fit_resample(X_train, y_train)
print("Distribución después de ADASYN:", ctr(y_train))

# --- Entrenar modelo SVM ---
svm_model = SVC(kernel='rbf', C=1, gamma=1, random_state=40)
svm_model.fit(X_train, y_train)

# --- Evaluación en validación ---
y_val_pred = svm_model.predict(X_val)
print("Accuracy en validación:", accuracy_score(y_val, y_val_pred))
print("Classification report (validación):\n", classification_report(y_val, y_val_pred))

# --- Matriz de confusión para la validación ---
conf_matrix = confusion_matrix(y_val, y_val_pred)
print("Matriz de Confusión (validación):\n", conf_matrix)

# --- Métricas adicionales para validación ---
precision = precision_score(y_val, y_val_pred, average='weighted')
recall = recall_score(y_val, y_val_pred, average='weighted')
f1 = f1_score(y_val, y_val_pred, average='weighted')

print(f"Precisión (weighted): {precision}")
print(f"Recall (weighted): {recall}")
print(f"F1-score (weighted): {f1}")

# --- Graficar la matriz de confusión ---
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Clase 0', 'Clase 1'], yticklabels=['Clase 0', 'Clase 1'])
plt.title("Matriz de Confusión (Validación)")
plt.xlabel("Predicciones")
plt.ylabel("Valores reales")
plt.show()

# ------------------ PRUEBA CON OTRO CONJUNTO DE DATOS ------------------
df_test = pd.read_csv("asthma_disease_data - copia.csv")  # <-- Asegúrate de tener este archivo
df_test = preprocess(df_test)

X_test_final = df_test[selected_features]
y_test_final = df_test['Diagnosis']

# --- Evaluación en conjunto externo ---
y_pred_final = svm_model.predict(X_test_final)
print("\nAccuracy en base de datos externa:", accuracy_score(y_test_final, y_pred_final))
print("Classification report (datos externos):\n", classification_report(y_test_final, y_pred_final))

# --- Matriz de confusión para el conjunto externo ---
conf_matrix_final = confusion_matrix(y_test_final, y_pred_final)
print("Matriz de Confusión (datos externos):\n", conf_matrix_final)

# --- Métricas adicionales para el conjunto externo ---
precision_final = precision_score(y_test_final, y_pred_final, average='weighted')
recall_final = recall_score(y_test_final, y_pred_final, average='weighted')
f1_final = f1_score(y_test_final, y_pred_final, average='weighted')

print(f"Precisión (weighted) en datos externos: {precision_final}")
print(f"Recall (weighted) en datos externos: {recall_final}")
print(f"F1-score (weighted) en datos externos: {f1_final}")

# --- Graficar la matriz de confusión para el conjunto externo ---
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_final, annot=True, fmt='d', cmap='Blues', xticklabels=['Clase 0', 'Clase 1'], yticklabels=['Clase 0', 'Clase 1'])
plt.title("Matriz de Confusión (Datos Externos)")
plt.xlabel("Predicciones")
plt.ylabel("Valores reales")
plt.show()

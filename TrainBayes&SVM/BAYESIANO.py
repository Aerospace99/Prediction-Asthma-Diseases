import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import ADASYN

# --- Preprocesamiento (idéntico) ---
def preprocess(df):
    df = df.copy()
    for col in ['Ethnicity', 'EducationLevel']:
        if col in df.columns:
            df[col] = df[col].replace(0, 4)

    binary_columns = []
    for col in df.columns:
        if col != 'Diagnosis' and col not in ['Ethnicity', 'EducationLevel']:
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).issubset({0, 1}):
                binary_columns.append(col)

    df[binary_columns] = df[binary_columns].replace(0, 2)

    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 19, 60, 100], labels=['Child', 'Teen', 'Adult', 'Senior'], right=False)
    df['BMICategory'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'], right=False)
    df['LifestyleScore'] = df[['PhysicalActivity', 'DietQuality', 'SleepQuality']].mean(axis=1)
    df['AllergyScore'] = df[['PetAllergy', 'HistoryOfAllergies', 'Eczema', 'HayFever']].sum(axis=1)
    df['ExposureScore'] = df[['PollutionExposure', 'PollenExposure', 'DustExposure']].mean(axis=1)
    df['SymptomSeverityScore'] = df[['Wheezing', 'ShortnessOfBreath', 'ChestTightness', 'Coughing', 'NighttimeSymptoms', 'ExerciseInduced']].sum(axis=1)
    df['LungFunctionRatio'] = df['LungFunctionFEV1'] / df['LungFunctionFVC']

    df = df.drop(columns=['PatientID', 'DoctorInCharge'], errors='ignore')
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

# --- Cargar y preparar datos ---
df = pd.read_csv("asthma_disease_data.csv")
df = preprocess(df)
X_all = df.drop('Diagnosis', axis=1)
y_all = df['Diagnosis']

# Selección de características
selector = SelectKBest(score_func=f_classif, k=min(15, X_all.shape[1]))
X_all = selector.fit_transform(X_all, y_all)

# Parámetros fijos
alpha = 0.001          # Valor para alpha
random_state = 86    # Valor para random_state

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=random_state)

# Escalado de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Balanceo de clases
X_train_bal, y_train_bal = ADASYN(random_state=random_state).fit_resample(X_train_scaled, y_train)

# Entrenamiento del modelo
model = BernoulliNB(alpha=alpha)
model.fit(X_train_bal, y_train_bal)

# Predicciones en el conjunto de test original
y_pred = model.predict(X_test_scaled)

# Evaluación del modelo con el conjunto original
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Resultados en el conjunto original
print("Evaluación con el dataset original:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Generar y mostrar la matriz de confusión para el conjunto original
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nMatriz de confusión para el dataset original:")
print(conf_matrix)

# --- Evaluación en un nuevo dataset ---

# Cargar y preprocesar el nuevo dataset
df_new = pd.read_csv("asthma_disease_data.csv")  # Aquí cargas el nuevo dataset
df_new = preprocess(df_new)
X_new = df_new.drop('Diagnosis', axis=1)
y_new = df_new['Diagnosis']

# Selección de características (con el mismo selector entrenado previamente)
X_new = selector.transform(X_new)

# Escalado de características usando el scaler entrenado
X_new_scaled = scaler.transform(X_new)

# Predicciones con el nuevo dataset
y_pred_new = model.predict(X_new_scaled)

# Evaluación en el nuevo dataset
accuracy_new = accuracy_score(y_new, y_pred_new)
precision_new = precision_score(y_new, y_pred_new)
recall_new = recall_score(y_new, y_pred_new)
f1_new = f1_score(y_new, y_pred_new)

# Resultados en el nuevo dataset
print("\nEvaluación con el nuevo dataset:")
print(f"Accuracy: {accuracy_new:.4f}")
print(f"Precision: {precision_new:.4f}")
print(f"Recall: {recall_new:.4f}")
print(f"F1 Score: {f1_new:.4f}")

# Generar y mostrar la matriz de confusión para el nuevo dataset
conf_matrix_new = confusion_matrix(y_new, y_pred_new)
print("\nMatriz de confusión para el nuevo dataset:")
print(conf_matrix_new)

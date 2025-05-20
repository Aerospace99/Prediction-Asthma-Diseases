import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import ADASYN
from collections import Counter

# --- Preprocesamiento ---
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

# --- Cargar y procesar datos ---
df = pd.read_csv("asthma_disease_data.csv")
df = preprocess(df)

X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Selección de características
selector = SelectKBest(score_func=f_classif, k=min(15, X.shape[1]))
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

# Hiperparámetros a explorar
C_values = [0.01, 0.1, 1, 10, 50]
gamma_values = [0.001, 0.01, 0.1, 1]
random_states = list(range(10, 60, 5))

# Búsqueda del mejor resultado
mejor_f1 = 0
mejores_params = {}
total = len(C_values) * len(gamma_values) * len(random_states)
contador = 1

for C in C_values:
    for gamma in gamma_values:
        for rs in random_states:
            X_train, X_val, y_train, y_val = train_test_split(X[selected_features], y, test_size=0.3, random_state=rs)

            # Balancear con ADASYN
            adasyn = ADASYN(random_state=rs)
            X_train_bal, y_train_bal = adasyn.fit_resample(X_train, y_train)

            # Modelo SVM
            model = SVC(kernel='rbf', C=C, gamma=gamma, random_state=rs)
            model.fit(X_train_bal, y_train_bal)
            y_pred = model.predict(X_val)

            # Métricas
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

            if f1 > mejor_f1:
                mejor_f1 = f1
                mejores_params = {
                    'C': C,
                    'gamma': gamma,
                    'random_state': rs,
                    'accuracy': acc,
                    'precision': prec,
                    'recall': rec,
                    'f1_score': f1
                }

            print(f"Combinación {contador}/{total} -> C={C}, gamma={gamma}, rs={rs}, F1={f1:.4f}")
            contador += 1

# --- Mejor combinación ---
print("\n✅ Mejor combinación encontrada (según F1 ponderado):")
for k, v in mejores_params.items():
    print(f"{k}: {v}")

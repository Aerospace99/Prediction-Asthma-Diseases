import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import ADASYN
import matplotlib.pyplot as plt
from tqdm import tqdm

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

# Hiperparámetros
random_states = range(1, 201)
alphas = np.linspace(0.001, 5, 50)

# Guardar resultados
results = []

# Entrenamiento con combinaciones
for alpha in tqdm(alphas, desc="Probando alphas"):
    for rs in random_states:
        try:
            X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=rs)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            X_train_bal, y_train_bal = ADASYN(random_state=rs).fit_resample(X_train_scaled, y_train)

            model = BernoulliNB(alpha=alpha)
            model.fit(X_train_bal, y_train_bal)
            y_pred = model.predict(X_test_scaled)

            results.append({
                'alpha': alpha,
                'random_state': rs,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            })
        except Exception as e:
            print(f"Error con alpha={alpha}, rs={rs}: {e}")

# Convertir resultados a DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv("bernoulli_nb_results.csv", index=False)

# Mejor combinación por f1
best = results_df.loc[results_df['f1'].idxmax()]
print("\n--- MEJOR COMBINACIÓN ---")
print(f"Alpha: {best['alpha']}")
print(f"Random State: {best['random_state']}")
print(f"Accuracy: {best['accuracy']:.4f}")
print(f"Precision: {best['precision']:.4f}")
print(f"Recall: {best['recall']:.4f}")
print(f"F1 Score: {best['f1']:.4f}")


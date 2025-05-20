import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from scipy.io import loadmat

# --- Cargar modelos y datos ---
svm_model = joblib.load('svm_model.pkl')
svm_selector = joblib.load('selector_main.pkl')
bayes_model = joblib.load('bayes_model.pkl')
bayes_selector = joblib.load('selector_main2.pkl')
bayes_scaler = joblib.load('bayes_scaler.pkl')

mat = loadmat('best_network.mat')
red = mat['red'][0, 0]
W1 = red['W1']
b1 = red['b1'].flatten()
W2 = red['W2']
b2 = red['b2'].flatten()
W3 = red['W3']
b3 = red['b3'].flatten()

minmax = loadmat('train_min_max.mat')
ValMin = minmax['ValMin'].flatten()
ValMax = minmax['ValMax'].flatten()
ymin = float(minmax.get('ymin', [[0.1]])[0][0])
ymax = float(minmax.get('ymax', [[0.9]])[0][0])

# --- Variables de entrada ---
campos = [
    'Edad', 'Género', 'Etnia', 'Nivel de Educacion', 'IMC', 'Fumador',
    'Actividad Fisica', 'Dieta', 'Calidad del sueño', 'Exposición a la contaminación',
    'Exposición al polen', 'Exposición al polvo', 'Alergia a las mascotas', 'Historial familiar con asma',
    'Historial de alergias', 'Eczema', 'Fiebre del heno', 'Reflujo gastroesofágico',
    'Función pulmonar FEV1', 'Función pulmonar FVC', 'Jadeos', 'Dificultad para respirar',
    'Opresión en el pecho', 'Tos', 'Síntomas nocturnos', 'Inducido por ejercicio'
]

default_values = [
    '30', '0', '2', '3', '22.5', '0', '5', '6', '7', '3', '4', '2', '0',
    '0', '0', '0', '0', '0', '3.2', '4.1', '0', '0', '0', '0', '0', '0'
]

entradas = {}

# --- Funciones compartidas ---
def obtener_datos_usuario():
    try:
        return {key: float(entry.get()) for key, entry in entradas.items()}
    except ValueError:
        raise ValueError("Por favor, ingresa valores válidos para todos los campos.")

def preprocess(df, for_model="svm"):
    df = df.copy()
    df['GrupoEdad'] = pd.cut(df['Edad'], bins=[0, 12, 19, 60, 100], labels=[0,1,2,3], right=False)
    df['CategoriaIMC'] = pd.cut(df['IMC'], bins=[0, 18.5, 24.9, 29.9, 100], labels=[0,1,2,3], right=False)
    df['PuntajeEstiloVida'] = df[['Actividad Fisica', 'Dieta', 'Calidad del sueño']].mean(axis=1)
    df['PuntajeAlergias'] = df[['Alergia a las mascotas', 'Historial de alergias', 'Eczema', 'Fiebre del heno']].sum(axis=1)
    df['PuntajeExposicion'] = df[['Exposición a la contaminación', 'Exposición al polen', 'Exposición al polvo']].mean(axis=1)
    df['PuntajeSintomas'] = df[['Jadeos', 'Dificultad para respirar', 'Opresión en el pecho', 'Tos', 'Síntomas nocturnos', 'Inducido por ejercicio']].sum(axis=1)
    df['RelacionPulmonar'] = df['Función pulmonar FEV1'] / df['Función pulmonar FVC']

    if for_model == "bayes":
        for col in ['Etnia', 'Nivel de Educacion']:
            df[col] = df[col].replace(0, 4)
        binary_columns = [col for col in df.columns if col not in ['Diagnostico', 'Etnia', 'Nivel de Educacion'] and set(df[col].dropna().unique()).issubset({0, 1})]
        df[binary_columns] = df[binary_columns].replace(0, 2)

    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    return df

# --- Clasificación SVM ---
def clasificar_svm():
    try:
        df = pd.DataFrame([obtener_datos_usuario()])
        prep = preprocess(df, for_model="svm")
        X = svm_selector.transform(prep.values)
        pred = svm_model.predict(X)[0]
        resultado_svm.set("SVM: Asmático" if pred == 1 else "SVM: No Asmático")
    except Exception as e:
        messagebox.showerror("Error SVM", str(e))

# --- Clasificación Bayes ---
def clasificar_bayes():
    try:
        df = pd.DataFrame([obtener_datos_usuario()])
        prep = preprocess(df, for_model="bayes")
        X = bayes_selector.transform(prep.values)
        X_scaled = bayes_scaler.transform(X)
        pred = bayes_model.predict(X_scaled)[0]
        resultado_bayes.set("Bayes: Asmático" if pred == 1 else "Bayes: No Asmático")
    except Exception as e:
        messagebox.showerror("Error Bayes", str(e))

# --- Clasificación Red Neuronal ---
def clasificar_red_neuronal():
    try:
        vals = [float(entradas[k].get()) for k in campos]

        Edad, IMC = vals[0], vals[4]
        ActFisica, Dieta, Sueño = vals[6], vals[7], vals[8]
        Mascotas, HistAlergias, Eczema, Heno = vals[12], vals[14], vals[15], vals[16]
        Contam, Polen, Polvo = vals[9], vals[10], vals[11]
        Jadeos, Respira, Pecho, Tos, Nocturnos, Ejercicio = vals[20:26]
        FEV1, FVC = vals[18], vals[19]

        GrupoEdad = 0 if Edad < 12 else 1 if Edad < 19 else 2 if Edad < 60 else 3
        CategoriaIMC = 0 if IMC < 18.5 else 1 if IMC < 25 else 2 if IMC < 30 else 3
        PuntajeEstilo = np.mean([ActFisica, Dieta, Sueño])
        PuntajeAlergia = np.sum([Mascotas, HistAlergias, Eczema, Heno])
        PuntajeExpo = np.mean([Contam, Polen, Polvo])
        PuntajeSintomas = np.sum([Jadeos, Respira, Pecho, Tos, Nocturnos, Ejercicio])
        RelPulmonar = 0 if FVC == 0 else FEV1 / FVC

        features = vals + [GrupoEdad, CategoriaIMC, PuntajeEstilo, PuntajeAlergia,
                           PuntajeExpo, PuntajeSintomas, RelPulmonar]

        features = np.array(features)
        denom = ValMax - ValMin
        denom[denom == 0] = 1
        norm = ((ymax - ymin) * (features - ValMin) / denom) + ymin

        A1 = np.tanh(W1 @ norm + b1)
        A2 = np.tanh(W2 @ A1 + b2)
        Z3 = W3 @ A2 + b3
        expZ = np.exp(Z3 - np.max(Z3))
        A3 = expZ / np.sum(expZ)

        pred = np.argmax(A3)
        resultado_red.set("Red Neuronal: Asmático" if pred == 1 else "Red Neuronal: No Asmático")
    except Exception as e:
        messagebox.showerror("Error Red Neuronal", str(e))

# --- Clasificación Total ---
def clasificar_todos():
    clasificar_svm()
    clasificar_bayes()
    clasificar_red_neuronal()

# --- GUI ---
ventana = tk.Tk()
ventana.title("Clasificador de Asma")

for idx, campo in enumerate(campos):
    tk.Label(ventana, text=campo+":", width=25).grid(row=idx, column=0, sticky=tk.W)
    entradas[campo] = tk.Entry(ventana)
    entradas[campo].insert(0, default_values[idx])
    entradas[campo].grid(row=idx, column=1)

tk.Button(ventana, text="Clasificar SVM", command=clasificar_svm).grid(row=len(campos), column=0, pady=5)
tk.Button(ventana, text="Clasificar Bayes", command=clasificar_bayes).grid(row=len(campos), column=1, pady=5)
tk.Button(ventana, text="Clasificar Red Neuronal", command=clasificar_red_neuronal).grid(row=len(campos)+1, column=0, columnspan=2, pady=5)
tk.Button(ventana, text="Clasificar Todos", command=clasificar_todos, bg="lightblue").grid(row=len(campos)+2, column=0, columnspan=2, pady=5)

resultado_svm = tk.StringVar()
resultado_bayes = tk.StringVar()
resultado_red = tk.StringVar()

tk.Label(ventana, textvariable=resultado_svm, fg="green").grid(row=len(campos)+3, column=0, columnspan=2)
tk.Label(ventana, textvariable=resultado_bayes, fg="blue").grid(row=len(campos)+4, column=0, columnspan=2)
tk.Label(ventana, textvariable=resultado_red, fg="purple").grid(row=len(campos)+5, column=0, columnspan=2)

ventana.mainloop()

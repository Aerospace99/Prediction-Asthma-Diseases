import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter 
from imblearn.over_sampling import SMOTE

# Cargar el conjunto de datos
df = pd.read_csv("asthma_disease_data.csv")

# Verificar datos faltantes
print("Datos faltantes por columna:")
print(df.isna().sum())

# Definir características y objetivo
features = df.drop(['PatientID', 'Diagnosis', 'DoctorInCharge'], axis=1)
target = df['Diagnosis']

# Aplicar SMOTE para balancear el conjunto de datos
smote = SMOTE(random_state=40)
X, y = features, target
X_resampled, y_resampled = smote.fit_resample(X, y)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=40)

# Definir y entrenar el modelo SVM
svm_model = SVC(kernel='rbf',C=1.0,gamma=0.5, max_iter=1000000000)
svm_model.fit(X_train, y_train)

# Realizar predicciones
y_pred_svm = svm_model.predict(X_test)

# Evaluar el modelo
accuracy_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm, average='binary')  # Cambiar a 'weighted' o 'macro' si es necesario

print(f"Support Vector Machine Accuracy: {accuracy_svm:.4f}")
print(f"Support Vector Machine F1 Score: {f1_svm:.4f}")

# Mostrar matriz de confusión
def display_confusion_matrix(y_true, y_pred, model_name):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Negative", "Positive"],
        cmap=plt.cm.Blues,
        colorbar=False
    )
    disp.ax_.set_title(f"{model_name} -- F1 Score: {f1_svm:.2f}")
    plt.show()

display_confusion_matrix(y_test, y_pred_svm, "Support Vector Machine")

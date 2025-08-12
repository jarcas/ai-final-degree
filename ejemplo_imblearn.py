from imblearn.over_sampling import SMOTE
import numpy as np

# Crear un conjunto de datos desequilibrado
X = np.random.rand(1000, 10)  # 1000 muestras, 10 características
y = np.zeros(1000)           # Todas las clases son 0 inicialmente
y[900:] = 1   

# Usando el mismo conjunto de datos del ejemplo anterior
smote = SMOTE(random_state=13)
X_smote, y_smote = smote.fit_resample(X, y)

print(f"Antes del resampling:")
print(f"Clase 0: {np.sum(y == 0)}")
print(f"Clase 1: {np.sum(y == 1)}")
print("\nDespués del SMOTE:")
print(f"Clase 0: {np.sum(y_smote == 0)}")
print(f"Clase 1: {np.sum(y_smote == 1)}")
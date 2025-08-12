import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Cargar dataset CICFlowMeter
df = pd.read_csv("20240625_Flooding_Heartbeat_filtered_ordered_labelled.csv")

# Convertir etiquetas a binario: 1 = ataque, 0 = normal
df['label_bin'] = df['Label'].apply(lambda x: 1 if 'cyberattack' in str(x).lower() else 0)

# Eliminar columnas no numéricas o identificadores
columns_to_drop = ['index', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Label']
X = df.drop(columns=columns_to_drop + ['label_bin'])
y = df['label_bin']

# Pipelines
pipeline_no_smote = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', KNeighborsClassifier())
])

pipeline_smote = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=13)),
    ('clf', KNeighborsClassifier())
])

# Validación cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
cm_no_smote = np.zeros((2, 2), dtype=int)
cm_smote = np.zeros((2, 2), dtype=int)

# Último modelo entrenado (para guardar)
final_no_smote_model = None
final_smote_model = None

for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Sin SMOTE
    pipeline_no_smote.fit(X_train, y_train)
    y_pred_no = pipeline_no_smote.predict(X_test)
    cm_no_smote += confusion_matrix(y_test, y_pred_no, labels=[0, 1])
    final_no_smote_model = pipeline_no_smote  # guardar el último modelo entrenado

    # Con SMOTE
    pipeline_smote.fit(X_train, y_train)
    y_pred_smote = pipeline_smote.predict(X_test)
    cm_smote += confusion_matrix(y_test, y_pred_smote, labels=[0, 1])
    final_smote_model = pipeline_smote  # guardar el último modelo entrenado

# Resultados
print("Matriz de confusión acumulada SIN SMOTE (KNN):")
print(cm_no_smote)

print("\nMatriz de confusión acumulada CON SMOTE (KNN):")
print(cm_smote)

# Guardar modelos con pickle
with open("modelo_knn_sin_smote.pkl", "wb") as f1:
    pickle.dump(final_no_smote_model, f1)

with open("modelo_knn_con_smote.pkl", "wb") as f2:
    pickle.dump(final_smote_model, f2)

print("\nModelos guardados como 'modelo_knn_sin_smote.pkl' y 'modelo_knn_con_smote.pkl'")

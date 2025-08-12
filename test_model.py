import pickle
import pandas as pd

# Cargar el modelo entrenado
with open("modelo_knn_con_smote.pkl", "rb") as f:
    modelo = pickle.load(f)

# Cargar nuevos datos
df_nuevo = pd.read_csv("20240625_Flooding_Heartbeat_filtered_ordered_labelled.csv")

# Si existe la columna 'Label' y quieres ver la predicción real vs esperada:
if 'Label' in df_nuevo.columns:
    df_nuevo['label_bin'] = df_nuevo['Label'].apply(lambda x: 1 if 'cyberattack' in str(x).lower() else 0)

# Columnas que deben eliminarse según entrenamiento previo
columns_to_drop = ['index', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Label']

# Eliminar columnas si existen
columns_to_drop = [col for col in columns_to_drop if col in df_nuevo.columns]
X_nuevo = df_nuevo.drop(columns=columns_to_drop)

# También elimina 'label_bin' si fue creada
if 'label_bin' in X_nuevo.columns:
    X_nuevo = X_nuevo.drop(columns=['label_bin'])

# Aplicar el modelo
y_pred = modelo.predict(X_nuevo)

# Mostrar primeras predicciones
print("Predicciones:", y_pred[:10])

# (Opcional) Guardar predicciones junto a los datos originales
df_nuevo['predicted_label'] = y_pred
df_nuevo.to_csv("test_con_predicciones.csv", index=False)

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Cargar el dataset
df = pd.read_csv("20240625_Flooding_Heartbeat_filtered_ordered_OcppFlows_120_labelled.csv")

# Crear columna binaria: 1 = ataque, 0 = normal
df['label_bin'] = df['label'].apply(lambda x: 1 if 'cyberattack' in str(x).lower() else 0)

# Eliminar columnas no numéricas o irrelevantes
cols_to_drop = [
    'index', 'flow_id', 'src_ip', 'dst_ip', 'src_port', 'dst_port',
    'flow_start_timestamp', 'flow_end_timestamp', 'label', 'label_bin'
]

# Variables independientes y etiqueta
X = df.drop(columns=cols_to_drop)
y = df['label_bin']

# Escalado de variables antes de PCA y SMOTE
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar SMOTE sobre datos escalados
X_resampled, y_resampled = SMOTE(random_state=13).fit_resample(X_scaled, y)

# PCA para visualizar en 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
X_resampled_pca = pca.transform(X_resampled)

# Graficar
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], label="Normal", alpha=0.6)
axes[0].scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], label="Ataque", alpha=0.6)
axes[0].set_title("Antes de SMOTE (escalado)")
axes[0].legend()
axes[0].grid(True)

axes[1].scatter(X_resampled_pca[y_resampled == 0, 0], X_resampled_pca[y_resampled == 0, 1], label="Normal (real + sintética)", alpha=0.6)
axes[1].scatter(X_resampled_pca[y_resampled == 1, 0], X_resampled_pca[y_resampled == 1, 1], label="Ataque", alpha=0.6)
axes[1].set_title("Después de SMOTE (escalado)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

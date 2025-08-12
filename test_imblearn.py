import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Cargar el dataset
filepath = "20240625_Flooding_Heartbeat_filtered_ordered_OcppFlows_120_labelled.csv"
df = pd.read_csv(filepath)

# Crear etiqueta binaria
df['label_bin'] = df['label'].apply(lambda x: 1 if 'cyberattack' in str(x).lower() else 0)

# Eliminar columnas irrelevantes
columns_to_drop = ['flow_id', 'flow_start_timestamp', 'flow_end_timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'label']
X = df.drop(columns=columns_to_drop + ['label_bin'])
y = df['label_bin']

# Validación cruzada y métrica
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
scoring = make_scorer(f1_score)

# 1️⃣ Modelo SIN SMOTE (sin balanceo)
pipeline_no_balance = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=13))
])
f1_no_balance = cross_val_score(pipeline_no_balance, X, y, cv=cv, scoring=scoring).mean()

# 2️⃣ Modelo CON SMOTE (con balanceo)
pipeline_with_smote = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=13)),
    ('clf', RandomForestClassifier(random_state=13))
])
f1_with_smote = cross_val_score(pipeline_with_smote, X, y, cv=cv, scoring=scoring).mean()

# Resultados comparativos
print("F1-score sin SMOTE:", round(f1_no_balance, 4))
print("F1-score con SMOTE:", round(f1_with_smote, 4))

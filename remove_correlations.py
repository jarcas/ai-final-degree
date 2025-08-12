import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score

# 1. Cargar datos
df = pd.read_csv("cleaned_dataset.csv")
df['flow_duration'] = df['flow_duration'] / 1e6
df['label_bin'] = df['label'].apply(lambda x: 1 if 'cyberattack' in str(x).lower() else 0)

# 2. Columnas irrelevantes
columns_to_drop = [
    'flow_id', 'flow_start_timestamp', 'flow_end_timestamp',
    'src_ip', 'dst_ip', 'src_port', 'dst_port', 'label'
]
existing_columns = [col for col in columns_to_drop if col in df.columns]

# 3. Crear X e y (con y sin flow_duration)
X_full = df.drop(columns=existing_columns + ['label_bin'])
y = df['label_bin']

# 4. Eliminar columnas con correlación >= 0.95
def drop_highly_correlated(df, threshold=0.95):
    corr_matrix = df.corr(numeric_only=True).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] >= threshold)]
    return df.drop(columns=to_drop), to_drop

X_reduced, dropped_cols = drop_highly_correlated(X_full)
print("Dropped highly correlated columns:", dropped_cols)

# 5. Crear variantes con y sin flow_duration
X_with = X_reduced.copy()
X_without = X_with.drop(columns=['flow_duration'], errors='ignore')

# 6. División train/test
Xw_train, Xw_test, yw_train, yw_test = train_test_split(X_with, y, stratify=y, test_size=0.2, random_state=13)
Xwo_train, Xwo_test, ywo_train, ywo_test = train_test_split(X_without, y, stratify=y, test_size=0.2, random_state=13)

# 7. Pipeline
def build_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=13))
    ])

param_grid = {
    'clf__n_estimators': [100],
    'clf__max_depth': [10],
    'clf__min_samples_split': [2],
    'clf__min_samples_leaf': [1],
    'clf__max_features': ['sqrt']
}

# 8. GridSearchCV y predicción
grid_with = GridSearchCV(build_pipeline(), param_grid, cv=StratifiedKFold(5), scoring='f1', n_jobs=-1)
grid_with.fit(Xw_train, yw_train)
yw_pred = grid_with.predict(Xw_test)

grid_without = GridSearchCV(build_pipeline(), param_grid, cv=StratifiedKFold(5), scoring='f1', n_jobs=-1)
grid_without.fit(Xwo_train, ywo_train)
ywo_pred = grid_without.predict(Xwo_test)

# 9. Resultados
results = pd.DataFrame({
    'Model': ['With flow_duration', 'Without flow_duration'],
    'Balanced Accuracy': [
        balanced_accuracy_score(yw_test, yw_pred),
        balanced_accuracy_score(ywo_test, ywo_pred)
    ],
    'F1 Score': [
        f1_score(yw_test, yw_pred),
        f1_score(ywo_test, ywo_pred)
    ],
    'ROC AUC': [
        roc_auc_score(yw_test, grid_with.predict_proba(Xw_test)[:, 1]),
        roc_auc_score(ywo_test, grid_without.predict_proba(Xwo_test)[:, 1])
    ]
})
print(results)

# 10. Visualización
metrics = ['Balanced Accuracy', 'F1 Score', 'ROC AUC']
x = range(len(metrics))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
for i, model in enumerate(results['Model']):
    ax.bar(
        [p + bar_width * i for p in x],
        results.iloc[i, 1:].values,
        bar_width,
        alpha=0.8,
        label=model
    )

ax.set_xlabel('Metric')
ax.set_ylabel('Score')
ax.set_title('Model Performance With vs Without flow_duration')
ax.set_xticks([p + bar_width / 2 for p in x])
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

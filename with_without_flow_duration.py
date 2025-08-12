import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar y preparar el dataset
df = pd.read_csv("cleaned_dataset.csv")
df['flow_duration'] = df['flow_duration'] / 1e6
df['label_bin'] = df['label'].apply(lambda x: 1 if 'cyberattack' in str(x).lower() else 0)

# 2. Eliminar columnas irrelevantes
columns_to_drop = [
    'flow_id', 'flow_start_timestamp', 'flow_end_timestamp',
    'src_ip', 'dst_ip', 'src_port', 'dst_port', 'label'
]
existing_columns = [col for col in columns_to_drop if col in df.columns]

# 3. Crear datasets
X_with = df.drop(columns=existing_columns + ['label_bin'])
X_without = X_with.drop(columns=['flow_duration'], errors='ignore')
y = df['label_bin']

# 4. Dividir en entrenamiento y test
Xw_train, Xw_test, yw_train, yw_test = train_test_split(X_with, y, stratify=y, test_size=0.2, random_state=13)
Xwo_train, Xwo_test, ywo_train, ywo_test = train_test_split(X_without, y, stratify=y, test_size=0.2, random_state=13)

# 5. Pipelines
def build_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=13))
    ])

def build_pipeline_with_smote():
    return ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=13)),
        ('clf', RandomForestClassifier(random_state=13))
    ])

# 6. Grid de hiperparámetros
param_grid = {
    'clf__n_estimators': [100],
    'clf__max_depth': [10],
    'clf__min_samples_split': [2],
    'clf__min_samples_leaf': [1],
    'clf__max_features': ['sqrt']
}

# Añadir parámetro para smote en GridSearch
param_grid_smote = {
    **param_grid,
    'smote__k_neighbors': [3]
}

# 7. Entrenamiento
grid_with = GridSearchCV(build_pipeline(), param_grid, cv=StratifiedKFold(5), scoring='f1', n_jobs=-1)
grid_with.fit(Xw_train, yw_train)
yw_pred = grid_with.predict(Xw_test)

grid_without = GridSearchCV(build_pipeline(), param_grid, cv=StratifiedKFold(5), scoring='f1', n_jobs=-1)
grid_without.fit(Xwo_train, ywo_train)
ywo_pred = grid_without.predict(Xwo_test)

grid_smote = GridSearchCV(build_pipeline_with_smote(), param_grid_smote, cv=StratifiedKFold(5), scoring='f1', n_jobs=-1)
grid_smote.fit(Xwo_train, ywo_train)
ywo_pred_smote = grid_smote.predict(Xwo_test)

# 8. Comparar resultados
results = pd.DataFrame({
    'Model': [
        'With flow_duration',
        'Without flow_duration',
        'Without flow_duration + SMOTE'
    ],
    'Balanced Accuracy': [
        balanced_accuracy_score(yw_test, yw_pred),
        balanced_accuracy_score(ywo_test, ywo_pred),
        balanced_accuracy_score(ywo_test, ywo_pred_smote)
    ],
    'F1 Score': [
        f1_score(yw_test, yw_pred),
        f1_score(ywo_test, ywo_pred),
        f1_score(ywo_test, ywo_pred_smote)
    ],
    'ROC AUC': [
        roc_auc_score(yw_test, grid_with.predict_proba(Xw_test)[:, 1]),
        roc_auc_score(ywo_test, grid_without.predict_proba(Xwo_test)[:, 1]),
        roc_auc_score(ywo_test, grid_smote.predict_proba(Xwo_test)[:, 1])
    ]
})

print(results)

# 9. Gráfica comparativa
metrics = ['Balanced Accuracy', 'F1 Score', 'ROC AUC']
x = range(len(metrics))
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.25

for i, model in enumerate(results['Model']):
    ax.bar(
        [p + bar_width * i for p in x],
        results.iloc[i, 1:].values,
        bar_width,
        alpha=0.8,
        label=model
    )

ax.set_xlabel('Metric', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks([p + bar_width for p in x])
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

importances = grid_with.best_estimator_.named_steps['clf'].feature_importances_
features = X_with.columns

# Crear DataFrame y ordenarlo
importances_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importances_df = importances_df.sort_values(by='Importance', ascending=False)

# Mostrar las más importantes
print(importances_df.head(10))

# Visualización
plt.figure(figsize=(10, 6))
sns.barplot(data=importances_df.head(10), x='Importance', y='Feature', palette='viridis')
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.show()

# Calcular matriz de correlación
corr_matrix = df.corr(numeric_only=True)

# Visualizar como mapa de calor
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False)
plt.title("Correlation Heatmap")
plt.show()

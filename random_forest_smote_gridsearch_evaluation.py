import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN

# Load dataset
# filepath = "20240625_Flooding_Heartbeat_filtered_ordered_OcppFlows_120_labelled.csv"
filepath = "cleaned_dataset.csv"
df = pd.read_csv(filepath)
# df['label_bin'] = df['label'].apply(lambda x: 1 if 'cyberattack' in str(x).lower() else 0)

# Drop irrelevant columns from the dataset
columns_to_drop = ['flow_id', 'flow_start_timestamp', 'flow_end_timestamp', 
                    'src_ip', 'dst_ip', 'src_port', 'dst_port', 'label', 'label_bin']

# Keep only columns that still exist in the DataFrame
columns_to_drop_existing = [col for col in columns_to_drop if col in df.columns]

# Create features and target
X = df.drop(columns=columns_to_drop_existing)
y = df['label_bin']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=13)

# Define pipelines
pipeline_no_smote = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=13))
])

pipeline_smote = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=13)),
    ('clf', RandomForestClassifier(random_state=13))
])

# pipeline_adasyn = ImbPipeline([
#     ('smote', ADASYN(random_state=13)),
#     ('scaler', StandardScaler()),
#     ('clf', RandomForestClassifier(random_state=13))
# ])

# Define parameter grid
param_grid = {
    'clf__n_estimators': [50, 100, 150],
    'clf__max_depth': [None, 10, 20, 30],
    'clf__min_samples_split': [2, 5, 10]
}

param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [10, 20, None],
    'clf__min_samples_split': [2, 5],
    'clf__min_samples_leaf': [1, 2],
    'clf__max_features': ['sqrt', 'log2']
}

# Perform grid search
grid_no_smote = GridSearchCV(pipeline_no_smote, param_grid, cv=StratifiedKFold(n_splits=5), scoring='f1', n_jobs=-1)
grid_smote = GridSearchCV(pipeline_smote, param_grid, cv=StratifiedKFold(n_splits=5), scoring='f1', n_jobs=-1)
# grid_adasyn = GridSearchCV(pipeline_adasyn, param_grid, cv=StratifiedKFold(n_splits=5), scoring='f1', n_jobs=-1)

# Fit models
grid_no_smote.fit(X_train, y_train)
grid_smote.fit(X_train, y_train)
# grid_adasyn.fit(X_train, y_train)

# Evaluate models
y_pred_no = grid_no_smote.predict(X_test)
y_pred_smote = grid_smote.predict(X_test)
# y_pred_adasyn = grid_adasyn.predict(X_test)

# Print results without SMOTE
print("Best params (without SMOTE):", grid_no_smote.best_params_)
print("Best score (without SMOTE):", grid_no_smote.best_score_)
print("Classification report (without SMOTE):")
print(classification_report(y_test, y_pred_no))

# Print results with SMOTE
print("Best params (with SMOTE):", grid_smote.best_params_)
print("Best score (with SMOTE):", grid_smote.best_score_)
print("Classification report (with SMOTE):")
print(classification_report(y_test, y_pred_smote))

# Print results with ADASYN
# print("Best params (with ADASYN):", grid_adasyn.best_params_)
# print("Best score (with ADASYN):", grid_adasyn.best_score_)
# print("Classification report (with ADASYN):")
# print(classification_report(y_test, y_pred_adasyn))


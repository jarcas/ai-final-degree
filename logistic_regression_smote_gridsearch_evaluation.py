import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Load dataset
filepath = "cleaned_dataset.csv"
df = pd.read_csv(filepath)

# Create binary label if not present
# df['label_bin'] = df['label'].apply(lambda x: 1 if 'cyberattack' in str(x).lower() else 0)

# Drop irrelevant columns
columns_to_drop = ['flow_id', 'flow_start_timestamp', 'flow_end_timestamp',
                   'src_ip', 'dst_ip', 'src_port', 'dst_port', 'label', 'label_bin']
columns_to_drop_existing = [col for col in columns_to_drop if col in df.columns]

# Create features and target
X = df.drop(columns=columns_to_drop_existing)
y = df['label_bin']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=13)

# Define pipelines
pipeline_no_smote = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=13))
])

pipeline_smote = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=13)),
    ('clf', LogisticRegression(max_iter=1000, random_state=13))
])

# Define parameter grid
param_grid = {
    'clf__C': [0.01, 0.1, 1, 10, 100],               
    'clf__penalty': ['l2'],                         
    'clf__solver': ['lbfgs']                        
}

# Grid search
grid_no_smote = GridSearchCV(pipeline_no_smote, param_grid,
                             cv=StratifiedKFold(n_splits=5),
                             scoring='f1', n_jobs=-1)

grid_smote = GridSearchCV(pipeline_smote, param_grid,
                          cv=StratifiedKFold(n_splits=5),
                          scoring='f1', n_jobs=-1)

# Fit models
grid_no_smote.fit(X_train, y_train)
grid_smote.fit(X_train, y_train)

# Predict
y_pred_no = grid_no_smote.predict(X_test)
y_pred_smote = grid_smote.predict(X_test)

# Results without SMOTE
print("Best params (without SMOTE):", grid_no_smote.best_params_)
print("Best F1 score (CV, no SMOTE):", grid_no_smote.best_score_)
print("Classification report (no SMOTE):")
print(classification_report(y_test, y_pred_no))

# Results with SMOTE
print("Best params (with SMOTE):", grid_smote.best_params_)
print("Best F1 score (CV, SMOTE):", grid_smote.best_score_)
print("Classification report (with SMOTE):")
print(classification_report(y_test, y_pred_smote))

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV  # For splitting data and hyperparameter tuning
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors classifier
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.pipeline import Pipeline  # To create machine learning pipelines
from sklearn.metrics import classification_report, confusion_matrix  # For evaluating the model
from imblearn.pipeline import Pipeline as ImbPipeline  # Importing imbalanced-learn pipeline for SMOTE
from imblearn.over_sampling import SMOTE  # Synthetic Minority Over-sampling Technique (SMOTE) for balancing classes

# Load the dataset from the CSV file
# filepath = "20240625_Flooding_Heartbeat_filtered_ordered_OcppFlows_120_labelled.csv"
filepath = "cleaned_dataset.csv"
df = pd.read_csv(filepath)

# Create a binary label column: 1 for 'cyberattack' and 0 for 'normal'
# df['label_bin'] = df['label'].apply(lambda x: 1 if 'cyberattack' in str(x).lower() else 0)

# Drop irrelevant columns from the dataset
columns_to_drop = ['flow_id', 'flow_start_timestamp', 'flow_end_timestamp', 
                    'src_ip', 'dst_ip', 'src_port', 'dst_port', 'label', 'label_bin']
# X = df.drop(columns=columns_to_drop + ['label_bin'])  # Features: drop irrelevant columns and binary label

# Keep only columns that still exist in the DataFrame
columns_to_drop_existing = [col for col in columns_to_drop if col in df.columns]

# Create features and target
X = df.drop(columns=columns_to_drop_existing)
y = df['label_bin']  # Target variable (y)

# Split the data into training and test sets (80% train, 20% test) while preserving the class distribution (stratification)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=13)

# Create a pipeline without SMOTE (StandardScaler + KNN classifier)
pipeline_no_smote = Pipeline([
    ('scaler', StandardScaler()),  # Standardize the features
    ('clf', KNeighborsClassifier())  # KNN classifier
])

# Create a pipeline with SMOTE (StandardScaler + SMOTE + KNN classifier)
pipeline_smote = ImbPipeline([
    ('scaler', StandardScaler()),  # Standardize the features
    ('smote', SMOTE(random_state=13)),  # Apply SMOTE to handle class imbalance
    ('clf', KNeighborsClassifier())  # KNN classifier
])

# Define the hyperparameters to search over for the KNN classifier
param_grid = {
    'clf__n_neighbors': [3, 5, 7, 9],  # Number of neighbors to use in KNN
    'clf__weights': ['uniform', 'distance'],  # Weighting strategy for the neighbors
    'clf__metric': ['euclidean', 'manhattan', 'minkowski']  # Distance metric for the KNN classifier
}

# Set up GridSearchCV for both pipelines (without and with SMOTE)
grid_no_smote = GridSearchCV(pipeline_no_smote, param_grid, cv=StratifiedKFold(n_splits=5), scoring='f1', n_jobs=-1)
grid_smote = GridSearchCV(pipeline_smote, param_grid, cv=StratifiedKFold(n_splits=5), scoring='f1', n_jobs=-1)

# Fit the models with GridSearchCV (training the models and finding the best hyperparameters)
grid_no_smote.fit(X_train, y_train)
grid_smote.fit(X_train, y_train)

# Make predictions on the test set
y_pred_no = grid_no_smote.predict(X_test)  # Predictions for the model without SMOTE
y_pred_smote = grid_smote.predict(X_test)  # Predictions for the model with SMOTE

# Print the best hyperparameters and the best score for the model without SMOTE
print("Best params (without SMOTE):", grid_no_smote.best_params_)
print("Best score (without SMOTE):", grid_no_smote.best_score_)

# Print the classification report for the model without SMOTE
print("Classification report (without SMOTE):")
print(classification_report(y_test, y_pred_no))  # Evaluate performance using classification metrics (precision, recall, F1-score)

# Print the best hyperparameters and the best score for the model with SMOTE
print("Best params (with SMOTE):", grid_smote.best_params_)
print("Best score (with SMOTE):", grid_smote.best_score_)

# Print the classification report for the model with SMOTE
print("Classification report (with SMOTE):")
print(classification_report(y_test, y_pred_smote))  # Evaluate performance using classification metrics (precision, recall, F1-score)

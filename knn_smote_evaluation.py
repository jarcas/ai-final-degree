# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold  # For cross-validation with stratification
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors classifier
from sklearn.preprocessing import StandardScaler  # For standardizing the features
from sklearn.pipeline import Pipeline  # To create machine learning pipelines
from sklearn.metrics import confusion_matrix, classification_report  # To evaluate the model performance
from imblearn.pipeline import Pipeline as ImbPipeline  # Importing imbalanced-learn pipeline for SMOTE
from imblearn.over_sampling import SMOTE  # Synthetic Minority Over-sampling Technique (SMOTE) to balance the dataset

# Load the dataset
# df = pd.read_csv("20240625_Flooding_Heartbeat_filtered_ordered_OcppFlows_120_labelled.csv")
df = pd. read_csv("cleaned_dataset.csv")

# Create a binary label: 1 for 'cyberattack' and 0 for 'normal'
# df['label_bin'] = df['label'].apply(lambda x: 1 if 'cyberattack' in str(x).lower() else 0)

# Drop irrelevant columns from the dataset
columns_to_drop = ['flow_id', 'flow_start_timestamp', 'flow_end_timestamp', 
                    'src_ip', 'dst_ip', 'src_port', 'dst_port', 'label', 'label_bin']
# X = df.drop(columns=columns_to_drop + ['label_bin'])  # Features: drop irrelevant columns and binary label

# Keep only columns that still exist in the DataFrame
columns_to_drop_existing = [col for col in columns_to_drop if col in df.columns]

# Create features and target
X = df.drop(columns=columns_to_drop_existing)
y = df['label_bin']  # Target variable: binary label

# Define a pipeline without SMOTE
pipeline_no_smote = Pipeline([
    ('scaler', StandardScaler()),  # Standardize the features
    ('clf', KNeighborsClassifier())  # KNN classifier
])

# Define a pipeline with SMOTE (to handle class imbalance)
pipeline_smote = ImbPipeline([
    ('scaler', StandardScaler()),  # Standardize the features
    ('smote', SMOTE(random_state=13)),  # Apply SMOTE to balance the classes
    ('clf', KNeighborsClassifier())  # KNN classifier
])

# Initialize Stratified K-Folds cross-validation with 5 splits
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)

# Initialize confusion matrices to accumulate results
cm_no_smote = np.zeros((2, 2), dtype=int)  # Confusion matrix for KNN without SMOTE
cm_smote = np.zeros((2, 2), dtype=int)  # Confusion matrix for KNN with SMOTE

# Perform cross-validation
for train_idx, test_idx in cv.split(X, y):  # Split the data into training and test sets
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]  # Split the features
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]  # Split the labels

    # Train and evaluate the model without SMOTE
    pipeline_no_smote.fit(X_train, y_train)  # Train the model
    y_pred_no = pipeline_no_smote.predict(X_test)  # Make predictions
    cm_no_smote += confusion_matrix(y_test, y_pred_no, labels=[0, 1])  # Update confusion matrix

    # Train and evaluate the model with SMOTE
    pipeline_smote.fit(X_train, y_train)  # Train the model with SMOTE
    y_pred_smote = pipeline_smote.predict(X_test)  # Make predictions
    cm_smote += confusion_matrix(y_test, y_pred_smote, labels=[0, 1])  # Update confusion matrix

# Print the cumulative confusion matrix for both models
print("Cumulative confusion matrix WITHOUT SMOTE (KNN):")
print(cm_no_smote)

print("\nCumulative confusion matrix WITH SMOTE (KNN):")
print(cm_smote)

# Print the classification report for the model without SMOTE
print("Classification report (without SMOTE):")
print(classification_report(y_test, y_pred_no))  # Evaluate performance using classification metrics (precision, recall, F1-score)

# Print the classification report for the model with SMOTE
print("Classification report (with SMOTE):")
print(classification_report(y_test, y_pred_smote))  # Evaluate performance using classification metrics (precision, recall, F1-score)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN

# Define the file path and load the dataset
# filepath = "20240625_Flooding_Heartbeat_filtered_ordered_OcppFlows_120_labelled.csv"
filepath = "cleaned_dataset.csv"
df = pd.read_csv(filepath)

# Create a binary label (1 for attack, 0 for normal) based on the 'label' column
# df['label_bin'] = df['label'].apply(lambda x: 1 if 'cyberattack' in str(x).lower() else 0)

# Drop irrelevant columns from the dataset
columns_to_drop = ['flow_id', 'flow_start_timestamp', 'flow_end_timestamp', 
                    'src_ip', 'dst_ip', 'src_port', 'dst_port', 'label', 'label_bin']
# X = df.drop(columns=columns_to_drop + ['label_bin'])  # Features: drop irrelevant columns and binary label

# Keep only columns that still exist in the DataFrame
columns_to_drop_existing = [col for col in columns_to_drop if col in df.columns]

# Create features and target
X = df.drop(columns=columns_to_drop_existing)
y = df['label_bin']  # Target variable (binary label)

# Create a pipeline without SMOTE (StandardScaler + RandomForestClassifier)
pipeline_no_smote = Pipeline([
    ('scaler', StandardScaler()),  # Standardize the features
    ('clf', RandomForestClassifier(random_state=13))  # Classifier
])

# Create a pipeline with SMOTE (StandardScaler + SMOTE + RandomForestClassifier)
pipeline_smote = ImbPipeline([
    ('scaler', StandardScaler()),  # Standardize the features
    ('smote', SMOTE(random_state=13)),  # Apply SMOTE to handle class imbalance
    ('clf', RandomForestClassifier(random_state=13))  # Classifier
])

# Create a pipeline with ADASYN (StandardScaler + ADASYN + RandomForestClassifier)
# pipeline_adasyn = ImbPipeline([
#     ('scaler', StandardScaler()),  # Standardize the features
#     ('adasyn', ADASYN(random_state=13)),  # Apply ADASYN to handle class imbalance
#     ('clf', RandomForestClassifier(random_state=13))  # Classifier
# ])

# Initialize Stratified K-Folds cross-validation with 5 splits
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)

# Initialize confusion matrices to store results
cm_no_smote = np.zeros((2, 2), dtype=int)  # Confusion matrix without SMOTE
cm_smote = np.zeros((2, 2), dtype=int)  # Confusion matrix with SMOTE
# cm_adasyn = np.zeros((2, 2), dtype=int)  # Confusion matrix with ADASYN

# Loop through each split of the cross-validation
for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]  # Split the data
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]  # Split the labels

    # Train and evaluate the model without SMOTE
    pipeline_no_smote.fit(X_train, y_train)  # Train the model
    y_pred_no = pipeline_no_smote.predict(X_test)  # Make predictions
    cm_no_smote += confusion_matrix(y_test, y_pred_no, labels=[0, 1])  # Update confusion matrix

    # Train and evaluate the model with SMOTE
    pipeline_smote.fit(X_train, y_train)  # Train the model with SMOTE
    y_pred_smote = pipeline_smote.predict(X_test)  # Make predictions
    cm_smote += confusion_matrix(y_test, y_pred_smote, labels=[0, 1])  # Update confusion matrix
    
    # Train and evaluate the model with ADASYN
    # pipeline_adasyn.fit(X_train, y_train)  # Train the model with ADASYN
    # y_pred_adasyn = pipeline_adasyn.predict(X_test)  # Make predictions
    # cm_adasyn += confusion_matrix(y_test, y_pred_adasyn, labels=[0, 1])

# Print confusion matrices for both cases (with and without SMOTE)
print("Confusion matrix WITHOUT SMOTE (Random Forest):")
print(cm_no_smote)

print("\nConfusion matrix WITH SMOTE (Random Forest):")
print(cm_smote)

# print("\nConfusion matrix WITH ADASYN (Random Forest):")
# print(cm_adasyn)

# Print the classification report for the model without SMOTE
print("Classification report (without SMOTE):")
print(classification_report(y_test, y_pred_no))  # Evaluate performance using classification metrics (precision, recall, F1-score)

# Print the classification report for the model with SMOTE
print("Classification report (with SMOTE):")
print(classification_report(y_test, y_pred_smote))  # Evaluate performance using classification metrics (precision, recall, F1-score)

# Print the classification report for the model with ADASYN
# print("Classification report (with ADASYN):")
# print(classification_report(y_test, y_pred_adasyn))
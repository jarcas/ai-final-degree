# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report

# Load training and test datasets
df_train = pd.read_csv("train_file_only_heartbeat.csv")
df_test = pd.read_csv("test_file_only_heartbeat.csv")

# Create binary labels: 1 for attack, 0 for normal
df_train['label_bin'] = df_train['label'].apply(lambda x: 1 if 'cyberattack' in str(x).lower() else 0)
df_test['label_bin'] = df_test['label'].apply(lambda x: 1 if 'cyberattack' in str(x).lower() else 0)

# Drop irrelevant columns
columns_to_drop = ['flow_id', 'flow_start_timestamp', 'flow_end_timestamp', 
                   'src_ip', 'dst_ip', 'src_port', 'dst_port', 'label']

X_train = df_train.drop(columns=columns_to_drop + ['label_bin'])
y_train = df_train['label_bin']

X_test = df_test.drop(columns=columns_to_drop + ['label_bin'])
y_test = df_test['label_bin']

# Define a pipeline with Random Forest (no need to scale features, but included for structure)
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Optional for RandomForest
    ('clf', RandomForestClassifier(random_state=13))  # Random Forest classifier
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on test set
y_pred = pipeline.predict(X_test)

# Evaluate model performance
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print("Confusion matrix:")
print(cm)

print("\nClassification report:")
print(classification_report(y_test, y_pred))

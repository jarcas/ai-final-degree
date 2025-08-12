
# Data manipulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc, confusion_matrix

# One-Class SVM model
from sklearn.svm import OneClassSVM

# Evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, f1_score

# For splitting and shuffling the data
from sklearn.model_selection import train_test_split


# General plot settings
sns.set_theme(style="whitegrid")


# Load dataset
# df = pd.read_csv('20240625_Flooding_Heartbeat_filtered_ordered_OcppFlows_120_labelled.csv')
df = pd.read_csv('cleaned_dataset.csv')

print("Dataset dimension:", df.shape)
df.head()


# Define binary target variable: 1 for attack, 0 for normal
y = df['label'].apply(lambda x: 1 if x == 'cyberattack_ocpp16_dos_flooding_heartbeat' else 0)

# Drop irrelevant columns from the dataset
columns_to_drop = ['flow_id', 'flow_start_timestamp', 'flow_end_timestamp', 
                    'src_ip', 'dst_ip', 'src_port', 'dst_port', 'label', 'label_bin']
# X = df.drop(columns=columns_to_drop + ['label_bin'])  # Features: drop irrelevant columns and binary label

# Keep only columns that still exist in the DataFrame
columns_to_drop_existing = [col for col in columns_to_drop if col in df.columns]

# Create features
X = df.drop(columns=columns_to_drop_existing)

# Show shape and class distribution
print("Feature matrix shape:", X.shape)
print("Label distribution:\n", y.value_counts())

# Select only attack and normal samples
X_attack = X[y == 1]
X_normal = X[y == 0]

# Split attack data into train (80%) and test (20%)
X_train, X_test_attack = train_test_split(X_attack, test_size=0.2, random_state=13)

# Use all normalimate data for testing
X_test_normal = X_normal.copy()

# Combine test sets
X_test = pd.concat([X_test_attack, X_test_normal])
y_test = [1] * len(X_test_attack) + [0] * len(X_test_normal)

# Check final shapes
print("Training set (attacks only):", X_train.shape)
print("Test set (attacks + normal):", X_test.shape)
print("Test label distribution:", pd.Series(y_test).value_counts())

# Initialize the One-Class SVM model
ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)

# Train the model on attack-only training data
ocsvm.fit(X_train)

print("Model trained on attack samples only.")


# Predict: +1 = inlier (attack), -1 = outlier (normal)
y_pred = ocsvm.predict(X_test)

# Map predictions to binary labels: 1 = attack, 0 = normal
y_pred_binary = [1 if p == 1 else 0 for p in y_pred]

# Classification report
print("=== Classification Report ===")
print(classification_report(y_test, y_pred_binary, target_names=['Normal', 'Attack']))

# Confusion matrix
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred_binary))

# ROC AUC score
roc_score = roc_auc_score(y_test, y_pred_binary)
print("ROC AUC Score:", roc_score)

# ROC Curve
y_scores = ocsvm.decision_function(X_test)  # raw anomaly scores
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – One-Class SVM")
plt.legend()
plt.grid(True)
plt.show()

# Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Normal', 'Predicted Attack'],
            yticklabels=['Actual Normal', 'Actual Attack'])
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.title("Confusion Matrix – One-Class SVM")
plt.show()



# Define parameter grid
nu_values = [0.01, 0.05, 0.1, 0.15, 0.2]
gamma_values = ['scale', 0.001, 0.01, 0.1]

# Store results
results = []

for nu in nu_values:
    for gamma in gamma_values:
        model = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
        model.fit(X_train)
        preds = model.predict(X_test)
        preds_binary = [1 if p == 1 else 0 for p in preds]

        f1 = f1_score(y_test, preds_binary)
        auc_score = roc_auc_score(y_test, preds_binary)

        results.append({
            'nu': nu,
            'gamma': gamma,
            'F1-score': f1,
            'ROC AUC': auc_score
        })

# Convert results to DataFrame for inspection
results_df = pd.DataFrame(results)
results_df.sort_values(by='F1-score', ascending=False)

# Pivot tables for heatmaps
f1_matrix = results_df.pivot(index='nu', columns='gamma', values='F1-score')
auc_matrix = results_df.pivot(index='nu', columns='gamma', values='ROC AUC')

# Plot F1-score heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(f1_matrix, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title("F1-score Heatmap (One-Class SVM)")
plt.ylabel("nu")
plt.xlabel("gamma")
plt.show()

# Plot ROC AUC heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(auc_matrix, annot=True, fmt=".3f", cmap="YlOrRd")
plt.title("ROC AUC Heatmap (One-Class SVM)")
plt.ylabel("nu")
plt.xlabel("gamma")
plt.show()

# Retrain with best hyperparameters
best_ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
best_ocsvm.fit(X_train)

# Predict and map to binary labels
y_pred = best_ocsvm.predict(X_test)
y_pred_binary = [1 if p == 1 else 0 for p in y_pred]

# Evaluation
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print("=== Classification Report ===")
print(classification_report(y_test, y_pred_binary, target_names=['Normal', 'Attack']))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred_binary))

roc_score = roc_auc_score(y_test, y_pred_binary)
print("ROC AUC Score:", roc_score)


# Compute anomaly scores
y_scores = best_ocsvm.decision_function(X_test)

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – One-Class SVM (nu=0.05, gamma='scale')")
plt.legend()
plt.grid(True)
plt.show()

# Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Normal', 'Predicted Attack'],
            yticklabels=['Actual Normal', 'Actual Attack'])
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.title("Confusion Matrix – One-Class SVM (nu=0.05, gamma='scale')")
plt.show()


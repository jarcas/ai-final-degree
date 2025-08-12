import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid")


# Load dataset
# df = pd.read_csv("20240625_Flooding_Heartbeat_filtered_ordered_OcppFlows_120_labelled.csv")
df = pd.read_csv("cleaned_dataset.csv")

# Convert label to binary: 1 = attack, 0 = normal
y = df['label'].apply(lambda x: 1 if x == 'cyberattack_ocpp16_dos_flooding_heartbeat' else 0)

# Drop irrelevant columns from the dataset
columns_to_drop = ['flow_id', 'flow_start_timestamp', 'flow_end_timestamp', 
                    'src_ip', 'dst_ip', 'src_port', 'dst_port', 'label', 'label_bin']
# X = df.drop(columns=columns_to_drop + ['label_bin'])  # Features: drop irrelevant columns and binary label

# Keep only columns that still exist in the DataFrame
columns_to_drop_existing = [col for col in columns_to_drop if col in df.columns]

# Create features and target
X = df.drop(columns=columns_to_drop_existing)

# Check shapes and distribution
print("Feature matrix shape:", X.shape)
print("Label distribution:\n", y.value_counts())


# Separate attack and normal samples
X_attack = X[y == 1]
X_normal = X[y == 0]

# Split attack samples: 80% for training, 20% for testing
X_train, X_test_attack = train_test_split(X_attack, test_size=0.2, random_state=13)

# Use all normal samples for test
X_test_normal = X_normal.copy()

# Combine test set
X_test = pd.concat([X_test_attack, X_test_normal])
y_test = [1] * len(X_test_attack) + [0] * len(X_test_normal)

# Confirm shapes
print("Training set (attacks only):", X_train.shape)
print("Test set (attacks + normal):", X_test.shape)
print("Test label distribution:", pd.Series(y_test).value_counts())


# Initialize and train the GMM
gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=13)
gmm.fit(X_train)

print("GMM trained on attack samples only.")

# Get log-likelihoods for training (to define threshold)
train_scores = gmm.score_samples(X_train)
threshold = np.percentile(train_scores, 5)  # 5th percentile

# Get log-likelihoods for test samples
test_scores = gmm.score_samples(X_test)

# Predict: 1 = attack (log-score above threshold), 0 = normal (below threshold)
y_pred = [1 if s > threshold else 0 for s in test_scores]

# Show threshold value
print(f"Log-likelihood threshold (5th percentile of training data): {threshold:.2f}")

# Convert y_test to Series for consistency
y_test_series = pd.Series(y_test)

# Classification report
print("=== Classification Report ===")
print(classification_report(y_test_series, y_pred, target_names=["Normal", "Attack"]))

# Confusion matrix
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test_series, y_pred))

# ROC AUC
roc_score = roc_auc_score(y_test_series, test_scores)
print(f"ROC AUC Score (using log-likelihoods): {roc_score:.4f}")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test_series, test_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Gaussian Mixture Model")
plt.legend()
plt.grid(True)
plt.show()

# Confusion matrix heatmap
cm = confusion_matrix(y_test_series, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Normal', 'Predicted Attack'],
            yticklabels=['Actual Normal', 'Actual Attack'])
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.title("Confusion Matrix – Gaussian Mixture Model")
plt.show()

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline as SkPipeline

# Configure plots
sns.set_theme(style="whitegrid")

# Load data
df_train = pd.read_csv("dataset_completo_client_1_2_3/Train1_cleaned.csv")
df_test = pd.read_csv("dataset_completo_client_1_2_3/Test1_cleaned.csv")

# Map labels to binary
df_train['label_bin'] = df_train['label'].apply(lambda x: 1 if str(x).lower().startswith('cyberattack') else 0)
df_test['label_bin'] = df_test['label'].apply(lambda x: 1 if str(x).lower().startswith('cyberattack') else 0)

# Show original class distribution
print("🔵 Original binary distribution in training set:")
print(df_train['label_bin'].value_counts())
print("\n🔵 Original binary distribution in test set:")
print(df_test['label_bin'].value_counts())

# Drop irrelevant columns
columns_to_drop = ['flow_id', 'flow_start_timestamp', 'flow_end_timestamp', 
                   'src_ip', 'dst_ip', 'src_port', 'dst_port', 'label']

X_train = df_train.drop(columns=columns_to_drop + ['label_bin'], errors='ignore')
y_train = df_train['label_bin']
X_test = df_test.drop(columns=columns_to_drop + ['label_bin'], errors='ignore')
y_test = df_test['label_bin']

# Ensure test set has the same feature structure and column order as training set.
# During cleaning, constant columns may be dropped independently in Train and Test files,
# which causes feature mismatches (e.g., a column present in Train but missing in Test).
# This reindexing guarantees that:
#   - X_test has exactly the same columns as X_train
#   - The column order is preserved (same as in X_train, not alphabetical)
#   - Any missing columns in X_test are filled with 0, which is a safe default for previously constant features.
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# ---------- 📊 Plot original class distribution ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(x=y_train, ax=ax[0])
ax[0].set_title("Before SMOTE (Train)")
ax[0].set_xlabel("Class")
ax[0].set_ylabel("Count")

# ---------- 🚫 Model without SMOTE ----------
pipeline_no_smote = SkPipeline([
    ('scaler', StandardScaler()),
    ('clf', KNeighborsClassifier())
])

pipeline_no_smote.fit(X_train, y_train)
y_pred_no_smote = pipeline_no_smote.predict(X_test)

report_no_smote = classification_report(y_test, y_pred_no_smote, output_dict=True)
print("\n🚫 WITHOUT SMOTE:")
print(confusion_matrix(y_test, y_pred_no_smote, labels=[0, 1]))
print(classification_report(y_test, y_pred_no_smote))

# ---------- ✅ Apply SMOTE manually (to show post-SMOTE distribution) ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
smote = SMOTE(random_state=13)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

# Plot resampled distribution
sns.countplot(x=y_resampled, ax=ax[1])
ax[1].set_title("After SMOTE (Train)")
ax[1].set_xlabel("Class")
ax[1].set_ylabel("Count")
plt.tight_layout()
plt.show()

# ---------- ✅ Model with SMOTE ----------
pipeline_smote = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=13)),
    ('clf', KNeighborsClassifier())
])

pipeline_smote.fit(X_train, y_train)
y_pred_smote = pipeline_smote.predict(X_test)

report_smote = classification_report(y_test, y_pred_smote, output_dict=True)
print("\n✅ WITH SMOTE:")
print(confusion_matrix(y_test, y_pred_smote, labels=[0, 1]))
print(classification_report(y_test, y_pred_smote))

# ---------- 🧩 Confusion Matrices (numeric + normalized), BEFORE and AFTER SMOTE ----------
# Compute confusion matrices
cm_no_smote = confusion_matrix(y_test, y_pred_no_smote, labels=[0, 1])
cm_smote = confusion_matrix(y_test, y_pred_smote, labels=[0, 1])

# Compute row-normalized versions (recall per class)
cm_no_smote_norm = cm_no_smote.astype(float) / cm_no_smote.sum(axis=1, keepdims=True)
cm_smote_norm = cm_smote.astype(float) / cm_smote.sum(axis=1, keepdims=True)
cm_no_smote_norm = np.nan_to_num(cm_no_smote_norm)
cm_smote_norm = np.nan_to_num(cm_smote_norm)

# Plot numeric confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_no_smote, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=[0, 1], yticklabels=[0, 1])
axes[0].set_title("Confusion Matrix - WITHOUT SMOTE (counts)")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=[0, 1], yticklabels=[0, 1])
axes[1].set_title("Confusion Matrix - WITH SMOTE (counts)")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# Plot normalized confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_no_smote_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[0],
            xticklabels=[0, 1], yticklabels=[0, 1], vmin=0, vmax=1)
axes[0].set_title("Confusion Matrix - WITHOUT SMOTE (row-normalized)")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(cm_smote_norm, annot=True, fmt='.2f', cmap='Greens', ax=axes[1],
            xticklabels=[0, 1], yticklabels=[0, 1], vmin=0, vmax=1)
axes[1].set_title("Confusion Matrix - WITH SMOTE (row-normalized)")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# ---------- 📋 Tabular Comparison of Metrics ----------
# 1. Extract per-class metrics
def extract_metrics(report, label):
    return {
        'Precision': report[str(label)]['precision'],
        'Recall': report[str(label)]['recall'],
        'F1-Score': report[str(label)]['f1-score'],
        'Support': report[str(label)]['support']
    }

# 2. Comparative DataFrames for class 0 (normal) and class 1 (attack)
metrics_class1_no_smote = extract_metrics(report_no_smote, 1)
metrics_class1_smote = extract_metrics(report_smote, 1)

metrics_class0_no_smote = extract_metrics(report_no_smote, 0)
metrics_class0_smote = extract_metrics(report_smote, 0)

df_class1 = pd.DataFrame({
    'WITHOUT SMOTE': metrics_class1_no_smote,
    'WITH SMOTE': metrics_class1_smote
})

df_class0 = pd.DataFrame({
    'WITHOUT SMOTE': metrics_class0_no_smote,
    'WITH SMOTE': metrics_class0_smote
})

# 3. Print tables
print("\n📊 COMPARISON TABLE (for class 1 - attack):")
print(df_class1.round(3))

print("\n📊 COMPARISON TABLE (for class 0 - normal):")
print(df_class0.round(3))

# 4. Prepare data for bar charts
def prepare_plot_df(df, class_label):
    df_plot = df.drop("Support")
    df_plot = df_plot.transpose()
    df_plot["Class"] = class_label
    df_plot["Metric"] = df_plot.index
    return df_plot.reset_index(drop=True)

df_plot_0 = prepare_plot_df(df_class0, "Class 0 (normal)")
df_plot_1 = prepare_plot_df(df_class1, "Class 1 (attack)")

df_plot = pd.concat([df_plot_0, df_plot_1], axis=0)

# 5. Bar charts per class
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for i, (class_label, df_sub) in enumerate(df_plot.groupby("Class")):
    df_melted = df_sub.melt(id_vars=["Metric", "Class"], var_name="SMOTE", value_name="Score")
    bar = sns.barplot(
        data=df_melted,
        x="Metric",
        y="Score",
        hue="SMOTE",
        ax=axes[i]
    )

    # Add value labels
    for container in bar.containers:
        bar.bar_label(container, fmt='%.3f', label_type='edge', fontsize=9, padding=2)

    axes[i].set_title(class_label)
    axes[i].set_ylim(0, 1.05)
    axes[i].legend(title="")

plt.tight_layout()
plt.show()

# ---------- 📐 Balanced Accuracy ----------
ba_no_smote = balanced_accuracy_score(y_test, y_pred_no_smote)
ba_smote = balanced_accuracy_score(y_test, y_pred_smote)

print("\n📐 Balanced Accuracy")
print(f"WITHOUT SMOTE: {ba_no_smote:.3f}")
print(f"WITH SMOTE   : {ba_smote:.3f}")

# ---------- 📈 ROC AUC (class 1 vs rest) ----------
# Use predict_proba to get scores
probs_no_smote = pipeline_no_smote.predict_proba(X_test)[:, 1]
probs_smote = pipeline_smote.predict_proba(X_test)[:, 1]

roc_auc_no_smote = roc_auc_score(y_test, probs_no_smote)
roc_auc_smote = roc_auc_score(y_test, probs_smote)

print("\n📈 ROC AUC (binary)")
print(f"WITHOUT SMOTE: {roc_auc_no_smote:.3f}")
print(f"WITH SMOTE   : {roc_auc_smote:.3f}")

# ---------- 📊 Comparison of Global Metrics (Bar Chart) ----------

# Accuracy general (opcional)
from sklearn.metrics import accuracy_score
acc_no_smote = accuracy_score(y_test, y_pred_no_smote)
acc_smote = accuracy_score(y_test, y_pred_smote)

# Create DataFrame for plotting
summary_metrics = pd.DataFrame({
    'WITHOUT SMOTE': {
        'Accuracy': acc_no_smote,
        'Balanced Accuracy': ba_no_smote,
        'ROC AUC': roc_auc_no_smote
    },
    'WITH SMOTE': {
        'Accuracy': acc_smote,
        'Balanced Accuracy': ba_smote,
        'ROC AUC': roc_auc_smote
    }
})

# Transpose for plotting
summary_metrics = summary_metrics.transpose().reset_index().rename(columns={"index": "Model"})

# Melt for seaborn
summary_melted = summary_metrics.melt(id_vars="Model", var_name="Metric", value_name="Score")

# Plot bar chart
fig, ax = plt.subplots(figsize=(8, 5))
barplot = sns.barplot(data=summary_melted, x="Metric", y="Score", hue="Model", ax=ax)

# Add value labels
for container in barplot.containers:
    barplot.bar_label(container, fmt='%.3f', label_type='edge', fontsize=9, padding=2)

ax.set_title("Global Performance Metrics Comparison")
ax.set_ylim(0.9, 1.01)
ax.set_ylabel("Score")
plt.tight_layout()
plt.show()


# ---------- 📋 Graphical Table: Class-wise Metrics ----------

def plot_metric_table(df, title):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')
    tbl = plt.table(cellText=df.round(3).values,
                    rowLabels=df.index,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.2)
    plt.tight_layout()
    plt.show()

# Plot tables for class 0 and class 1
plot_metric_table(df_class0, "Class 0 (Normal) - Metrics Comparison")
plot_metric_table(df_class1, "Class 1 (Attack) - Metrics Comparison")

# Plot table for summary metrics
plot_metric_table(summary_metrics.set_index("Model").round(3), "Global Metrics Comparison")

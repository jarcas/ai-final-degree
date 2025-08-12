# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    accuracy_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline as SkPipeline

# Load cleaned datasets
df_train = pd.read_csv("dataset_completo_client_1_2_3/Train1_cleaned.csv")
df_test = pd.read_csv("dataset_completo_client_1_2_3/Test1_cleaned.csv")

# Create binary labels
df_train['label_bin'] = df_train['label'].apply(lambda x: 1 if str(x).lower().startswith("cyberattack") else 0)
df_test['label_bin'] = df_test['label'].apply(lambda x: 1 if str(x).lower().startswith("cyberattack") else 0)

# Drop irrelevant columns
columns_to_drop = ['flow_id', 'flow_start_timestamp', 'flow_end_timestamp',
                   'src_ip', 'dst_ip', 'src_port', 'dst_port', 'label']

X_test = df_test.drop(columns=columns_to_drop + ['label_bin'], errors='ignore')
y_test = df_test['label_bin']

# Ensure X_test structure matches
# base_X_train = df_train.drop(columns=columns_to_drop + ['label_bin'], errors='ignore')
# X_test = X_test.reindex(columns=base_X_train.columns, fill_value=0)

# Define percentages of class 0 to keep
percentages = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05]
results = []

for pct in percentages:
    if pct == 1.0:
        # Use the full training dataset without modifying order
        df_reduced = df_train.copy()
    else:
        # Reduce class 0
        df_0 = df_train[df_train['label_bin'] == 0]
        df_1 = df_train[df_train['label_bin'] == 1]
        n_samples = int(len(df_0) * pct)
        df_0_down = df_0.sample(n=n_samples, random_state=13)
        df_reduced = pd.concat([df_1, df_0_down]).sample(frac=1, random_state=13)

    # Train/test split
    X_train = df_reduced.drop(columns=columns_to_drop + ['label_bin'], errors='ignore')
    y_train = df_reduced['label_bin']
    X_test_aligned = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Train without SMOTE
    pipe_no_smote = SkPipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier())
    ])
    pipe_no_smote.fit(X_train, y_train)
    y_pred_ns = pipe_no_smote.predict(X_test_aligned)
    y_score_ns = pipe_no_smote.predict_proba(X_test_aligned)[:, 1]

    ba_ns = balanced_accuracy_score(y_test, y_pred_ns)
    roc_ns = roc_auc_score(y_test, y_score_ns)
    f1_ns = f1_score(y_test, y_pred_ns)

    # Train with SMOTE
    pipe_smote = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=13)),
        ('clf', KNeighborsClassifier())
    ])
    pipe_smote.fit(X_train, y_train)
    y_pred_s = pipe_smote.predict(X_test_aligned)
    y_score_s = pipe_smote.predict_proba(X_test_aligned)[:, 1]

    ba_s = balanced_accuracy_score(y_test, y_pred_s)
    roc_s = roc_auc_score(y_test, y_score_s)
    f1_s = f1_score(y_test, y_pred_s)
    precision_ns = precision_score(y_test, y_pred_ns, pos_label=0)
    precision_s = precision_score(y_test, y_pred_s, pos_label=0)
    acc_ns = accuracy_score(y_test, y_pred_ns)
    acc_s = accuracy_score(y_test, y_pred_s)
    f1_class_0_ns = f1_score(y_test, y_pred_ns, pos_label=0)
    f1_class_0_s = f1_score(y_test, y_pred_s, pos_label=0)

    results.append({
        '% class 0 in train': int(pct * 100),
        'Balanced Accuracy (No SMOTE)': ba_ns,
        'Balanced Accuracy (With SMOTE)': ba_s,
        'ROC AUC (No SMOTE)': roc_ns,
        'ROC AUC (With SMOTE)': roc_s,
        'Accuracy (No SMOTE)': acc_ns,
        'Accuracy (With SMOTE)': acc_s,
        'Precision (No SMOTE)': precision_ns,
        'Precision (With SMOTE)': precision_s,
        'F1-score (No SMOTE)': f1_ns,
        'F1-score (With SMOTE)': f1_s,
        'F1-class0 (No SMOTE)': f1_class_0_ns,
        'F1-class0 (With SMOTE)': f1_class_0_s
    })
    
# Create DataFrame
results_df = pd.DataFrame(results)

# ---------- 📊 Plot Balanced Accuracy with annotations ----------
# sns.set_theme(style="whitegrid")
# plt.figure(figsize=(10, 6))
# ax = plt.gca()
# sns.lineplot(data=results_df, x='% class 0 in train', y='Balanced Accuracy (No SMOTE)', marker='o', label='Without SMOTE', ax=ax)
# sns.lineplot(data=results_df, x='% class 0 in train', y='Balanced Accuracy (With SMOTE)', marker='s', label='With SMOTE', ax=ax)

# for i, row in results_df.iterrows():
#     ax.text(row['% class 0 in train'], row['Balanced Accuracy (No SMOTE)'] + 0.005,
#             f"{row['Balanced Accuracy (No SMOTE)']:.3f}", ha='center', fontsize=9, color='black')
#     ax.text(row['% class 0 in train'], row['Balanced Accuracy (With SMOTE)'] - 0.025,
#             f"{row['Balanced Accuracy (With SMOTE)']:.3f}", ha='center', fontsize=9, color='black')

# plt.title("SMOTE Effect on Balanced Accuracy vs Class 0 Reduction")
# plt.xlabel("% of class 0 samples used in training")
# plt.ylabel("Balanced Accuracy on test set")
# plt.ylim(0.4, 1.05)
# plt.legend()
# plt.tight_layout()
# plt.show()

# ---------- 📊 Plot Balanced Accuracy with updated X-axis labels ----------
# Calculate the actual number of class 0 samples in the original training dataset
total_class_0 = df_train[df_train['label_bin'] == 0].shape[0]

# Add column with the actual number of class 0 samples used at each percentage
results_df['# class 0 samples'] = (results_df['% class 0 in train'] / 100 * total_class_0).astype(int)

# Create combined label for X-axis with both percentage and sample count
results_df['Train Class 0'] = results_df['% class 0 in train'].astype(str) + "% (" + results_df['# class 0 samples'].astype(str) + ")"

# Sort DataFrame for plotting
results_df = results_df.sort_values(by='% class 0 in train', ascending=True).reset_index(drop=True)

# plt.figure(figsize=(10, 6))
# ax = plt.gca()

# sns.lineplot(data=results_df, x='Train Class 0', y='Balanced Accuracy (No SMOTE)', marker='o', label='Without SMOTE', ax=ax)
# sns.lineplot(data=results_df, x='Train Class 0', y='Balanced Accuracy (With SMOTE)', marker='s', label='With SMOTE', ax=ax)

# for i, row in results_df.iterrows():
#     ax.text(i, row['Balanced Accuracy (No SMOTE)'] - 0.025, f"{row['Balanced Accuracy (No SMOTE)']:.3f}",
#             ha='center', fontsize=8, color='black')
#     ax.text(i, row['Balanced Accuracy (With SMOTE)'] + 0.025, f"{row['Balanced Accuracy (With SMOTE)']:.3f}",
#             ha='center', fontsize=8, color='black')

# plt.title("SMOTE Effect on Balanced Accuracy")
# plt.xlabel("% of class 0 samples used in training (real count in parentheses)")
# plt.ylabel("Balanced Accuracy")
# plt.ylim(0.4, 1.05)
# plt.xticks(rotation=30)
# plt.legend()
# plt.tight_layout()
# plt.show()

# ---------- 💾 Export results to CSV ----------
results_df.to_csv("evaluate_smote_vs_imbalance.csv", index=False)

# Calculate the actual number of class 0 samples in the original training dataset
# total_class_0 = df_train[df_train['label_bin'] == 0].shape[0]

# Sort the DataFrame by % class 0 in train
# results_df = results_df.sort_values(by='% class 0 in train', ascending=True).reset_index(drop=True)

# Add column with the actual number of class 0 samples used at each percentage
# results_df['# class 0 samples'] = (results_df['% class 0 in train'] / 100 * total_class_0).astype(int)

# Create combined label for X-axis with both percentage and sample count
# results_df['Train Class 0'] = results_df['% class 0 in train'].astype(str) + "% (" + results_df['# class 0 samples'].astype(str) + ")"

# Sort X-axis values by percentage for consistent plotting
# results_df = results_df.sort_values(by='% class 0 in train', ascending=True)

# --------- PLOTTING FUNCTIONS ---------
def plot_metric(metric_col_1, metric_col_2, ylabel, title):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    sns.lineplot(data=results_df, x='Train Class 0', y=metric_col_1, marker='o', label='Without SMOTE', ax=ax)
    sns.lineplot(data=results_df, x='Train Class 0', y=metric_col_2, marker='s', label='With SMOTE', ax=ax)

    # Annotate values on the lines
    for i, row in results_df.iterrows():
        ax.text(i, row[metric_col_1] - 0.025, f"{row[metric_col_1]:.3f}", ha='center', fontsize=8, color='black')
        ax.text(i, row[metric_col_2] + 0.025, f"{row[metric_col_2]:.3f}", ha='center', fontsize=8, color='black')

    plt.title(title)
    plt.xlabel("% of class 0 samples used in training (real count in parentheses)")
    plt.ylabel(ylabel)
    plt.ylim(0.4, 1.05)
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --------- PLOT METRICS ---------
plot_metric("Balanced Accuracy (No SMOTE)", "Balanced Accuracy (With SMOTE)",
            "Balanced Accuracy", "SMOTE Effect on Balanced Accuracy")

plot_metric("F1-score (No SMOTE)", "F1-score (With SMOTE)",
            "F1 Score (Class 1)", "SMOTE Effect on F1 Score - Class 1")

plot_metric("ROC AUC (No SMOTE)", "ROC AUC (With SMOTE)",
            "ROC AUC", "SMOTE Effect on ROC AUC")

plot_metric("F1-class0 (No SMOTE)", "F1-class0 (With SMOTE)",
            "F1 Score (Class 0)", "SMOTE Effect on F1 Score - Class 0")

plot_metric("Accuracy (No SMOTE)", "Accuracy (With SMOTE)",
            "Accuracy", "SMOTE Effect on Accuracy")

plot_metric("Precision (No SMOTE)", "Precision (With SMOTE)",
            "Precision (Class 0)", "SMOTE Effect on Precision - Class 0")

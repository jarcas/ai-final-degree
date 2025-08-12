import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import argparse
import os
import numpy as np

def detect_iqr_outliers(series):
    """Detect outliers in a series using the IQR method."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (series < lower) | (series > upper)

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Clean and analyze a labeled network dataset.")
    parser.add_argument("input_file", help="Path to the input CSV file")
    args = parser.parse_args()

    file_path = args.input_file
    df = pd.read_csv(file_path)

    # Create output filename
    base, ext = os.path.splitext(file_path)
    output_file = f"{base}_cleaned{ext}"

    # Convert duration from µs to s
    df['flow_duration'] = df['flow_duration'] / 1e6

    print("Number of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])

    print("\nClass distribution:")
    print(df['label'].value_counts(dropna=False))

    print("\nMissing values per column:")
    print(df.isnull().sum())
        
    # Print number of rows before dropping NaNs
    print(f"\nRows before dropping NaNs: {df.shape[0]}")
    # Drop any rows with missing values
    df = df.dropna()
    # Print number of rows after dropping NaNs
    print(f"Rows after dropping NaNs : {df.shape[0]}")

    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    print("\nConstant columns (no variability):")
    print(constant_cols)

    columns_to_remove = ['index', 'flow_id', 'src_ip', 'dst_ip', 'src_port', 'dst_port']
    print("\nSuggested columns to drop (identifiers or irrelevant):")
    print(columns_to_remove)

    print("\n--- Timestamp Column Evaluation ---")
    timestamp_cols = ['flow_start_timestamp', 'flow_end_timestamp']
    for col in timestamp_cols:
        if col in df.columns:
            print(f"\n{col}:")
            print(f"  min: {df[col].min():.0f}")
            print(f"  max: {df[col].max():.0f}")
            print(f"  mean: {df[col].mean():.0f}")
            print(f"  std: {df[col].std():.0f}")
            print(f"  range: {df[col].max() - df[col].min():.0f}")
            print(f"  is_monotonic: {df[col].is_monotonic_increasing}")

    df['label_bin'] = df['label'].apply(lambda x: 1 if 'cyberattack' in str(x).lower() else 0)
    if 'flow_duration' in df.columns:
        print(f"\nCorrelation with flow_duration:")
        print(f"  start_timestamp: {df['flow_start_timestamp'].corr(df['flow_duration']):.4f}")
        print(f"  end_timestamp:   {df['flow_end_timestamp'].corr(df['flow_duration']):.4f}")
        print(f"\nCorrelation with label_bin:")
        print(f"  start_timestamp: {df['flow_start_timestamp'].corr(df['label_bin']):.4f}")
        print(f"  end_timestamp:   {df['flow_end_timestamp'].corr(df['label_bin']):.4f}")

    columns_to_remove += ['flow_start_timestamp', 'flow_end_timestamp']

    num_duplicates = df.duplicated().sum()
    print(f"\nNumber of duplicated rows: {num_duplicates}")
    if num_duplicates > 0:
        print("\nDuplicated rows preview:")
        print(df[df.duplicated()].head())

    zero_only_cols = [col for col in df.columns if (df[col] == 0).all()]
    print("\nColumns with only zeros:")
    print(zero_only_cols)

    all_to_drop = list(set(columns_to_remove + constant_cols + zero_only_cols))
    df_cleaned = df.drop(columns=all_to_drop, errors='ignore')
    print(f"\nFinal shape after dropping: {df_cleaned.shape}")

    # Remove inf/-inf values
    df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
    num_inf_rows = df_cleaned.isna().sum().sum()
    if num_inf_rows > 0:
        print(f"\nRows with inf or -inf (converted to NaN): {num_inf_rows}")
        df_cleaned.dropna(inplace=True)
        print(f"Rows after removing inf/-inf values: {df_cleaned.shape[0]}")
    
    df_cleaned.to_csv(output_file, index=False)
    print(f"\nFinal cleaned dataset saved to: {output_file}")

    print("\nRunning outlier detection...")

    df_cleaned['label_bin'] = df_cleaned['label'].apply(lambda x: 1 if 'cyberattack' in str(x).lower() else 0)
    X = df_cleaned.drop(columns=['label', 'label_bin'])

    iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=13)
    df_cleaned['outlier'] = pd.Series(iso.fit_predict(X)).map({1: 0, -1: 1})
    print(f"Number of outliers detected: {df_cleaned['outlier'].sum()}")

    top2 = X.var().sort_values(ascending=False).head(2).index.tolist()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=df_cleaned[top2[0]],
        y=df_cleaned[top2[1]],
        hue=df_cleaned['outlier'],
        palette={0: 'blue', 1: 'red'},
        alpha=0.6
    )
    plt.title("Outlier Detection using Isolation Forest")
    plt.xlabel(top2[0])
    plt.ylabel(top2[1])
    plt.legend(title="Outlier", labels=["Inlier", "Outlier"])
    plt.tight_layout()
    plt.show()

    print("\nPlotting boxplots for numeric features...")
    numeric_cols = df_cleaned.select_dtypes(include=['number']).drop(columns=['label_bin', 'outlier'], errors='ignore').columns.tolist()
    X_melted = df_cleaned[numeric_cols].melt(var_name="Feature", value_name="Value")

    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Feature", y="Value", data=X_melted)
    plt.xticks(rotation=45, ha='right')
    plt.title("Boxplots of Numerical Features")
    plt.tight_layout()
    plt.show()

    print(f"Selected features for outlier analysis: {top2}")

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_cleaned, x='label_bin', y='flow_duration', palette='Set2')
    plt.title('Flow Duration by Class')
    plt.xlabel('Class (0 = Normal, 1 = Cyberattack)')
    plt.ylabel('Flow Duration (seconds)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    for feature in top2:
        df_cleaned[f"{feature}_outlier"] = detect_iqr_outliers(df[feature])
        plt.figure(figsize=(8, 5))
        sns.scatterplot(
            data=df_cleaned,
            x='flow_duration',
            y=feature,
            hue=df_cleaned[f"{feature}_outlier"],
            palette={False: 'blue', True: 'red'},
            alpha=0.7
        )
        plt.title(f"Outliers in {feature} vs flow_duration")
        plt.xlabel("flow_duration")
        plt.ylabel(feature)
        plt.legend(title="Outlier")
        plt.tight_layout()
        plt.show()

    outlier_analysis = {}
    for feature in top2:
        outlier_mask = detect_iqr_outliers(df_cleaned[feature])
        normal = df_cleaned.loc[outlier_mask & (df_cleaned["label_bin"] == 0), "flow_duration"]
        attack = df_cleaned.loc[outlier_mask & (df_cleaned["label_bin"] == 1), "flow_duration"]
        outlier_analysis[feature] = {
            "normal_mean": normal.mean(),
            "normal_count": len(normal),
            "attack_mean": attack.mean(),
            "attack_count": len(attack)
        }

    summary_data = []
    for feature, values in outlier_analysis.items():
        summary_data.append({
            "Feature": feature,
            "Normal Count": values["normal_count"],
            "Normal Mean flow_duration (s)": round(values["normal_mean"], 2),
            "Attack Count": values["attack_count"],
            "Attack Mean flow_duration (s)": round(values["attack_mean"], 2)
        })

    outlier_summary_df = pd.DataFrame(summary_data)
    print(outlier_summary_df.to_string(index=False))

    plot_data, count_data = [], []
    for _, row in outlier_summary_df.iterrows():
        plot_data += [
            {"Feature": row["Feature"], "Class": "Normal", "Mean Flow Duration (s)": row["Normal Mean flow_duration (s)"]},
            {"Feature": row["Feature"], "Class": "Attack", "Mean Flow Duration (s)": row["Attack Mean flow_duration (s)"]}
        ]
        count_data += [
            {"Feature": row["Feature"], "Class": "Normal", "Outlier Count": row["Normal Count"]},
            {"Feature": row["Feature"], "Class": "Attack", "Outlier Count": row["Attack Count"]}
        ]

    plot_df = pd.DataFrame(plot_data)
    count_df = pd.DataFrame(count_data)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    ax1 = sns.barplot(data=plot_df, x="Feature", y="Mean Flow Duration (s)", hue="Class", ax=axes[0])
    axes[0].set_title("Mean Flow Duration of Outliers")
    axes[0].set_ylabel("Mean Flow Duration (s)")
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1f', label_type='edge', fontsize=10)

    ax2 = sns.barplot(data=count_df, x="Feature", y="Outlier Count", hue="Class", ax=axes[1])
    axes[1].set_title("Number of Outliers")
    axes[1].set_ylabel("Outlier Count")
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%d', label_type='edge', fontsize=10)

    plt.tight_layout()
    plt.show()

    # outlier_columns = ['outlier'] + [f"{feature}_outlier" for feature in top2]
    # df_cleaned = df_cleaned.drop(columns=outlier_columns, errors='ignore')
    # df_cleaned.to_csv(output_file, index=False)
    # print(f"\nFinal cleaned dataset saved to: {output_file}")

if __name__ == "__main__":
    main()

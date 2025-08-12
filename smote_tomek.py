import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                           balanced_accuracy_score, roc_auc_score,
                           precision_recall_curve, roc_curve, f1_score)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data(filepath):
    """Load and explore the initial dataset"""
    print("=" * 60)
    print("LOADING AND EXPLORING DATASET")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv(filepath)
    print(f"Dataset dimensions: {df.shape}")

    # Create binary label
    # df['label_bin'] = df['label'].apply(lambda x: 1 if 'cyberattack' in str(x).lower() else 0)

    # Analyze class distribution
    print(f"\nOriginal class distribution:")
    original_dist = df['label'].value_counts()
    print(original_dist)

    print(f"\nBinary class distribution:")
    binary_dist = df['label_bin'].value_counts()
    print(binary_dist)

    # Calculate imbalance ratio
    ratio = binary_dist[0] / binary_dist[1] if len(binary_dist) > 1 else "No minority class"
    print(f"Imbalance ratio: {ratio:.2f}:1" if isinstance(ratio, float) else ratio)

    # Visualize class distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Original labels
    axes[0].pie(binary_dist.values, labels=['Normal', 'Cyberattack'],
                autopct='%1.1f%%', colors=['#2E8B57', '#DC143C'])
    axes[0].set_title('Class Distribution\n(Original Dataset)')

    # Bar plot
    axes[1].bar(['Normal', 'Cyberattack'], binary_dist.values,
                color=['#2E8B57', '#DC143C'], alpha=0.7)
    axes[1].set_title('Count by Class')
    axes[1].set_ylabel('Number of Samples')

    # Add value labels on bars
    for i, v in enumerate(binary_dist.values):
        axes[1].text(i, v + max(binary_dist.values) * 0.01, str(v),
                    ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Data quality check
    print(f"\nData quality:")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")

    return df

def preprocess_data(df):
    """Preprocess data by removing irrelevant columns"""
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    # Drop irrelevant columns
    columns_to_drop = ['flow_id', 'flow_start_timestamp', 'flow_end_timestamp',
                      'src_ip', 'dst_ip', 'src_port', 'dst_port', 'label']

    # Check which columns actually exist
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    print(f"Columns to drop: {existing_columns_to_drop}")

    X = df.drop(columns=existing_columns_to_drop + ['label_bin'])
    y = df['label_bin']

    print(f"Final features: {X.shape[1]} columns")
    print(f"Samples: {X.shape[0]}")

    # Check for any remaining non-numeric columns
    non_numeric = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        print(f"⚠️  Non-numeric columns detected: {list(non_numeric)}")
        # Convert to numeric or handle as needed
        for col in non_numeric:
            X[col] = pd.to_numeric(X[col], errors='coerce')

    # Handle missing values
    if X.isnull().sum().sum() > 0:
        print("Filling missing values with median...")
        X = X.fillna(X.median())

    return X, y

def create_pipelines():
    """Create improved pipelines for comparison"""
    print("\n" + "=" * 60)
    print("CREATING IMPROVED PIPELINES")
    print("=" * 60)

    # Pipeline without balancing (with class_weight)
    pipeline_no_smote = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=13, class_weight='balanced'))
    ])

    # Pipeline with SMOTE (corrected order)
    pipeline_smote = ImbPipeline([
        ('scaler', StandardScaler()),  # ✅ FIRST: Scale the data
        ('smote', SMOTE(random_state=13, k_neighbors=5)),  # SECOND: Balance classes
        ('clf', RandomForestClassifier(random_state=13))  # THIRD: Classify
    ])

    # Pipeline with ADASYN (corrected order)
    # pipeline_adasyn = ImbPipeline([
    #     ('scaler', StandardScaler()),
    #     ('adasyn', ADASYN(random_state=13, n_neighbors=5)),
    #     ('clf', RandomForestClassifier(random_state=13))
    # ])

    # Pipeline with SMOTETomek (hybrid approach)
    pipeline_smotetomek = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smotetomek', SMOTETomek(smote=SMOTE(random_state=13), random_state=13)),
        ('clf', RandomForestClassifier(random_state=13))
    ])

    pipelines = {
        'No Balancing': pipeline_no_smote,
        'SMOTE': pipeline_smote,
        # 'ADASYN': pipeline_adasyn,
        'SMOTE+Tomek': pipeline_smotetomek
    }

    print("✅ Pipelines created:")
    for name in pipelines.keys():
        print(f"  - {name}")

    return pipelines

def define_param_grid():
    """Define optimized parameter grid"""
    print("\n" + "=" * 60)
    print("DEFINING PARAMETER GRID")
    print("=" * 60)

    # Base parameters for Random Forest
    base_params = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [10, 20, None],
        'clf__min_samples_split': [2, 5],
        'clf__min_samples_leaf': [1, 2],
        'clf__max_features': ['sqrt', 'log2']
    }

    # Specific parameters for each pipeline
    param_grids = {
        'No Balancing': base_params,
        'SMOTE': {
            **base_params,
            'smote__k_neighbors': [3, 5]
        },
        # 'ADASYN': {
        #     **base_params,
        #     'adasyn__n_neighbors': [3, 5]
        # },
        'SMOTE+Tomek': {
            **base_params,
            'smotetomek__smote__k_neighbors': [3, 5]
        }
    }

    print("✅ Parameter grids defined")
    return param_grids

def train_and_evaluate_models(pipelines, param_grids, X_train, X_test, y_train, y_test):
    """Train and evaluate all models with GridSearchCV"""
    print("\n" + "=" * 60)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 60)

    results = {}
    best_models = {}

    # Configure multiple scoring
    scoring = ['balanced_accuracy', 'f1', 'precision', 'recall', 'roc_auc']

    for name, pipeline in pipelines.items():
        print(f"\n🔄 Training: {name}")
        print("-" * 40)

        # GridSearchCV with multiple metrics
        grid_search = GridSearchCV(
            pipeline,
            param_grids[name],
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=13),
            scoring=scoring,
            refit='balanced_accuracy',  # Primary metric for selection
            n_jobs=-1,
            verbose=0
        )

        # Train
        grid_search.fit(X_train, y_train)

        # Predictions
        y_pred = grid_search.predict(X_test)
        y_pred_proba = grid_search.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'best_params': grid_search.best_params_,
            'cv_balanced_accuracy': grid_search.best_score_,
            'test_balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'test_f1': f1_score(y_test, y_pred),
            'test_auc_roc': roc_auc_score(y_test, y_pred_proba),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }

        results[name] = metrics
        best_models[name] = grid_search.best_estimator_

        # Show results
        print(f"✅ Best parameters: {grid_search.best_params_}")
        print(f"📊 CV Balanced Accuracy: {grid_search.best_score_:.4f}")
        print(f"📊 Test Balanced Accuracy: {metrics['test_balanced_accuracy']:.4f}")
        print(f"📊 Test F1-Score: {metrics['test_f1']:.4f}")
        print(f"📊 Test AUC-ROC: {metrics['test_auc_roc']:.4f}")

    return results, best_models

def visualize_results_old(results, y_test):
    """Create comprehensive visualizations of results"""
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # 1. Metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    models = list(results.keys())
    metrics_comparison = {
        'Balanced Accuracy': [results[model]['test_balanced_accuracy'] for model in models],
        'F1-Score': [results[model]['test_f1'] for model in models],
        'AUC-ROC': [results[model]['test_auc_roc'] for model in models]
    }

    # Bar plots for metrics
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    for i, (metric, values) in enumerate(metrics_comparison.items()):
        row, col = i // 2, i % 2
        bars = axes[row, col].bar(models, values, color=colors, alpha=0.8)
        axes[row, col].set_title(f'{metric} by Model', fontsize=14, fontweight='bold')
        axes[row, col].set_ylabel(metric)
        axes[row, col].set_ylim(0, 1)

        # Add values on bars
        for bar, value in zip(bars, values):
            axes[row, col].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        # Rotate labels if necessary
        axes[row, col].tick_params(axis='x', rotation=45)

    # Summary table in remaining subplot
    axes[1, 1].axis('off')
    table_data = []
    for model in models:
        table_data.append([
            model,
            f"{results[model]['test_balanced_accuracy']:.3f}",
            f"{results[model]['test_f1']:.3f}",
            f"{results[model]['test_auc_roc']:.3f}"
        ])

    table = axes[1, 1].table(cellText=table_data,
                            colLabels=['Model', 'Bal. Acc.', 'F1', 'AUC-ROC'],
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title('Metrics Summary', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # 2. Confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (name, result) in enumerate(results.items()):
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
        axes[i].set_title(f'Confusion Matrix - {name}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    plt.tight_layout()
    plt.show()

    # 3. ROC curves
    plt.figure(figsize=(12, 8))

    for i, (name, result) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        auc_score = result['test_auc_roc']
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})',
                linewidth=2, color=colors[i])

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.show()

def visualize_results(results, y_test):
    """Create comprehensive visualizations of results"""
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve

    models = list(results.keys())
    metrics_comparison = {
        'Balanced Accuracy': [results[model]['test_balanced_accuracy'] for model in models],
        'F1-Score': [results[model]['test_f1'] for model in models],
        'AUC-ROC': [results[model]['test_auc_roc'] for model in models]
    }

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    # Create custom grid layout: 1 row for 3 metrics, 1 row for summary table
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Model Evaluation Metrics', fontsize=16, fontweight='bold')

    # GridSpec for custom layout
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 3, height_ratios=[3, 1])

    axes = []
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        metric = list(metrics_comparison.keys())[i]
        values = metrics_comparison[metric]

        bars = ax.bar(models, values, color=colors[:len(models)], alpha=0.8)
        ax.set_title(f'{metric} by Model', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)

        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        ax.tick_params(axis='x', rotation=45)
        axes.append(ax)

    # Summary table below the bar charts (spans all 3 columns)
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')

    table_data = []
    for model in models:
        table_data.append([
            model,
            f"{results[model]['test_balanced_accuracy']:.3f}",
            f"{results[model]['test_f1']:.3f}",
            f"{results[model]['test_auc_roc']:.3f}"
        ])

    table = ax_table.table(cellText=table_data,
                           colLabels=['Model', 'Bal. Acc.', 'F1', 'AUC-ROC'],
                           cellLoc='center',
                           loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # 2. Confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (name, result) in enumerate(results.items()):
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
        axes[i].set_title(f'Confusion Matrix - {name}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    plt.tight_layout()
    plt.show()

    # 3. ROC curves
    plt.figure(figsize=(12, 8))

    for i, (name, result) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        auc_score = result['test_auc_roc']
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})',
                 linewidth=2, color=colors[i])

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.show()

def print_detailed_results(results):
    """Print detailed results for each model"""
    print("\n" + "=" * 60)
    print("DETAILED RESULTS")
    print("=" * 60)

    for name, result in results.items():
        print(f"\n🔍 {name.upper()}")
        print("-" * 50)
        print(f"Best parameters:")
        for param, value in result['best_params'].items():
            print(f"  {param}: {value}")

        print(f"\nPerformance metrics:")
        print(f"  CV Balanced Accuracy: {result['cv_balanced_accuracy']:.4f}")
        print(f"  Test Balanced Accuracy: {result['test_balanced_accuracy']:.4f}")
        print(f"  Test F1-Score: {result['test_f1']:.4f}")
        print(f"  Test AUC-ROC: {result['test_auc_roc']:.4f}")

        print(f"\nClassification report:")
        print(result['classification_report'])

def main():
    """Main function that executes the entire pipeline"""
    # Define filepath
    # filepath = "20240625_Flooding_Heartbeat_filtered_ordered_OcppFlows_120_labelled.csv"
    filepath = "cleaned_dataset.csv"

    try:
        # 1. Load and explore data
        df = load_and_explore_data(filepath)

        # 2. Preprocess data
        X, y = preprocess_data(df)

        # 3. Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=13
        )

        print(f"\nData split:")
        print(f"  Training: {X_train.shape}")
        print(f"  Testing: {X_test.shape}")

        # 4. Create pipelines
        pipelines = create_pipelines()

        # 5. Define parameters
        param_grids = define_param_grid()

        # 6. Train and evaluate
        results, best_models = train_and_evaluate_models(
            pipelines, param_grids, X_train, X_test, y_train, y_test
        )

        # 7. Visualize results
        visualize_results(results, y_test)

        # 8. Print detailed results
        print_detailed_results(results)

        # 9. Final recommendation
        best_model_name = max(results.keys(),
                            key=lambda x: results[x]['test_balanced_accuracy'])
        print(f"\n🏆 RECOMMENDED MODEL: {best_model_name}")
        print(f"   Balanced Accuracy: {results[best_model_name]['test_balanced_accuracy']:.4f}")

        return results, best_models

    except FileNotFoundError:
        print(f"❌ Error: File {filepath} not found")
        print("   Please verify the file path.")
        return None, None
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        return None, None

# Execute complete analysis
if __name__ == "__main__":
    results, models = main()

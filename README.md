# Evaluation of Resampling Methods on Imbalanced OCPP Dataset

This repository contains a Jupyter notebook and associated data/results for evaluating different **resampling strategies** to handle class imbalance in Intrusion Detection Systems (IDS) applied to **Electric Vehicle (EV) charging infrastructures**. The dataset is based on OCPP 1.6 traffic flows preprocessed with **OCPPFlowMeter**.

---

## Repository Structure


### Results folder (`results/`)
This folder contains the outputs produced by the notebook:
- **CSV files** with evaluation metrics  
  (`resampling_methods_results.csv`, `mean_metrics_by_technique.csv`, etc.)
- **Plots** illustrating performance across resampling techniques  
  (`accuracy-resampling_effects.png`, `balanced_accuracy_subplots-resampling_effects.png`, `roc_auc-resampling_effects.png`, etc.)

---

## Notebook Overview

The notebook implements the following workflow:

1. **Dataset preparation**  
   - Load OCPP 1.6 traffic datasets (`Train1.csv`, `Test1.csv`)  
   - Clean features (remove constant/zero-only columns, convert durations, etc.)  
   - Save cleaned datasets (`Train1_cleaned.csv`, `Test1_cleaned.csv`)

2. **Resampling methods evaluated**  
   - **Oversampling**: SMOTE, ADASYN, SMOTE+ENN, SMOTE+Tomek  
   - **Undersampling**: Random Undersampling (RUS), NearMiss v1/v2/v3  

3. **Classifier**  
   - Fixed hyperparameter **k-Nearest Neighbors (kNN)** model

4. **Evaluation metrics**  
   - Accuracy  
   - Balanced Accuracy  
   - Precision, Recall, and F1-score (per class)  
   - ROC AUC  

5. **Visualization**  
   - Subplots by resampling technique  
   - Recall/F1 comparisons across imbalance levels  
   - Export of plots and CSV summaries to the `results/` folder  

---

## Installation

Clone the repository and install dependencies:

```bash
git clone <your_repo_url>
cd <your_repo_name>
pip install -r requirements.txt
```

## Usage

Run the Jupyter notebook:

```bash
jupyter notebook resampling_methods.ipynb
```

## Purpose

The main goal of this work is to **quantify the impact of different resampling methods** on classifier performance under severe class imbalance. The insights are valuable for designing robust IDS in EV charging infrastructures.


## License

This repository is provided for academic and research purposes.  
Please cite appropriately if used in your work.

from utils.ml_models_pipeline import run_coxph, run_logreg, run_xgboost, run_svm, run_mlpclassifier, run_random_forest, run_rsf
from utils.neural_focal_loss import run_nn_focal
import numpy as np
import pandas as pd
import tensorflow as tf

import random
import os

from utils.support_functions import summarize_cv_metrics

import warnings
warnings.filterwarnings("ignore")

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

def main(dataset_path="data/df_normal.csv", dataset_name="Normal Dataset", time_var="survTime", status_var="statusComp"):

    ## 5-fold cross-validation outputs
    print(f"Running models on dataset: {dataset_name}")
    df = pd.read_csv(dataset_path)

    neural_focal_loss = run_nn_focal(df, time_var=time_var, status_var=status_var) # Neural Network with Focal Loss
    logreg_results = run_logreg(df, time_var=time_var, status_var=status_var) # Logistic Regression
    rf_results = run_random_forest(df, time_var=time_var, status_var=status_var) # Random Forest
    xgb_results = run_xgboost(df, time_var=time_var, status_var=status_var) # XGBoost
    svm_results = run_svm(df, time_var=time_var, status_var=status_var) # SVM
    result_mlp = run_mlpclassifier(df, time_var=time_var, status_var=status_var) # Neural Network
    rsf_results = run_rsf(df, duration_col=time_var, event_col=status_var) # Random Survival Forest
    cox_results = run_coxph(df, duration_col=time_var, event_col=status_var) # Cox Proportional Hazards model

    model_results_list = [
        ("CoxPH", cox_results), # Cox Proportional Hazards model
        ("NNet", result_mlp), # Neural Network
        ("LogReg", logreg_results), # Logistic Regression
        ("XGBoost", xgb_results), # XGBoost
        ("SVM", svm_results), # Support Vector Machine
        ("RF", rf_results), # Random Forest
        ("RSF", rsf_results), # Random Survival Forest
        ("NNetFocal", neural_focal_loss), # Neural Network with Focal Loss
    ]

    summary_df = summarize_cv_metrics(model_results_list, dataset_name="PBC Dataset")
    print(summary_df)

    summary_df = summary_df.reset_index()

    summary_df.to_csv(f"output/{dataset_name}_results.csv", index=False)

#main(dataset_path="data/df_normal.csv", dataset_name="PBC_dataset", time_var="survTime", status_var="statusComp")

main(dataset_path="data/TCGA_dataset/clinical/df_white_kirc.csv", dataset_name="clinical_white_kirc", time_var="OS.time", status_var="OS") 

#main(dataset_path="data/TCGA_dataset/mirna/df_white_kirc.csv", dataset_name="mirna_white_kirc", time_var="OS.time", status_var="OS") 

#main(dataset_path="data/TCGA_dataset/rnaseq/df_white_kirc.csv", dataset_name="rnaseq_white_kirc", time_var="OS.time", status_var="OS") 

# main(dataset_path="data/TCGA_dataset/mirna/df_black_kirc.csv", dataset_name="mirna_black_kirc", time_var="OS.time", status_var="OS") 

# main(dataset_path="data/TCGA_dataset/rnaseq/df_black_kirc.csv", dataset_name="rnaseq_black_kirc", time_var="OS.time", status_var="OS") 

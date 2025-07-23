from utils.ml_models_pipeline import run_coxph, run_logreg, run_xgboost, run_svm, run_mlpclassifier, run_random_forest, run_rsf
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

    logreg_results = run_logreg(df, time_var=time_var, status_var=status_var) #logistic regression
    rf_results = run_random_forest(df, time_var=time_var, status_var=status_var) #random forest
    xgb_results = run_xgboost(df, time_var=time_var, status_var=status_var) #xgboost
    svm_results = run_svm(df, time_var=time_var, status_var=status_var) #svm
    result_mlp = run_mlpclassifier(df, time_var=time_var, status_var=status_var) #neural network
    rsf_results = run_rsf(df, duration_col=time_var, event_col=status_var) #random survival forest
    cox_results = run_coxph(df, duration_col=time_var, event_col=status_var) # Cox Proportional Hazards model

    #print(cox_results)

    model_results_list = [
        ("CoxPH", cox_results), # Cox Proportional Hazards model
        ("NNet", result_mlp), # Neural Network
        ("LogReg", logreg_results), # Logistic Regression
        ("XGBoost", xgb_results), # XGBoost
        ("SVM", svm_results), # Support Vector Machine
        ("RF", rf_results), # Random Forest
        ("RSF", rsf_results) # Random Survival Forest
    ]

    summary_df = summarize_cv_metrics(model_results_list, dataset_name="PBC Dataset")
    print(summary_df)

    summary_df = summary_df.reset_index()

    summary_df.to_csv(f"output/{dataset_name}_results.csv", index=False)

#main(dataset_path="data/df_normal.csv", dataset_name="PBC_dataset", time_var="survTime", status_var="statusComp")

#main(dataset_path="data/TCGA_dataset/clinical/df_white_kirc.csv", dataset_name="clinical_white_kirc", time_var="OS.time", status_var="OS") ## convergence error for coxph

main(dataset_path="data/TCGA_dataset/mirna/df_white_kirc.csv", dataset_name="mirna_white_kirc", time_var="OS.time", status_var="OS") ## convergence error for coxph

main(dataset_path="data/TCGA_dataset/rnaseq/df_white_kirc.csv", dataset_name="rnaseq_white_kirc", time_var="OS.time", status_var="OS") ## convergence error for coxph


# main(dataset_path="data/TCGA_dataset/mirna/df_black_kirc.csv", dataset_name="mirna_black_kirc", time_var="OS.time", status_var="OS") ## convergence error for coxph

# main(dataset_path="data/TCGA_dataset/rnaseq/df_black_kirc.csv", dataset_name="rnaseq_black_kirc", time_var="OS.time", status_var="OS") ## convergence error for coxph

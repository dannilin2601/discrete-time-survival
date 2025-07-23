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

def main(dataset_path="data/df_normal.csv", dataset_name="Normal Dataset"):

    ## 5-fold cross-validation outputs
    print(f"Running models on dataset: {dataset_name}")
    df = pd.read_csv(dataset_path)

    cox_results = run_coxph(df) # Cox Proportional Hazards model
    logreg_results = run_logreg(df) #logistic regression
    rf_results = run_random_forest(df) #random forest
    xgb_results = run_xgboost(df) #xgboost
    svm_results = run_svm(df) #svm
    result_mlp = run_mlpclassifier(df) #neural network
    rsf_results = run_rsf(df) #random survival forest

    # print(rsf_results)
    # print(cox_results)

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

main(dataset_path="data/df_normal.csv", dataset_name="PBC_dataset")
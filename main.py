from utils.ml_models_pipeline import run_nnet, run_coxph
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

    print(f"Running models on dataset: {dataset_name}")
    df = pd.read_csv(dataset_path)

    nnet_results = run_nnet(df)
    cox_results = run_coxph(df)

    model_results_list = [
        ("NeuralNet", nnet_results),
        ("CoxPH", cox_results)
    ]

    summary_df = summarize_cv_metrics(model_results_list, dataset_name="PBC Dataset")
    print(summary_df)

    summary_df.to_csv(f"output/{dataset_name}_results.csv", index=False)

main(dataset_path="data/df_normal.csv", dataset_name="PBC_datset")
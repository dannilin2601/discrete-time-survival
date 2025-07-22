import os
import random
import csv

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import tensorflow as tf

from skopt import gp_minimize
from skopt.space import Integer, Real

from person_period import generate_person_time_data

##### Neural Network Functions #####

def fit_nnet(X, y, hidden_layers=(32,), alpha=0.0001, activation='relu'):
    """
    Train a neural network (MLPClassifier) with configurable architecture.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target labels
    hidden_layers : tuple
        Tuple defining the number of neurons in each hidden layer (e.g., (64,), (128, 64))
    alpha : float
        L2 regularization strength (default=0.0001)
    activation : str
        Activation function: 'relu', 'tanh', or 'logistic' (default='relu')

    Returns
    -------
    model : MLPClassifier
        Trained MLPClassifier model
    """
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,  # supports multiple layers
        alpha=alpha,                       # L2 regularization
        activation=activation,             # relu / tanh / logistic
        max_iter=500,
        solver='adam',
        random_state=42
    )
    model.fit(X, y)
    return model


def evaluate_metrics(y_true, y_prob):
    """Compute Brier, ROC AUC, and PR-AUC for given predictions."""
    brier = brier_score_loss(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    return brier, auc, pr_auc

from sklearn.model_selection import StratifiedKFold
import numpy as np

def nnet_cv(df, time_var="survTime", status_var="statusComp", hidden_layers=(32,), alpha=0.0001, bins=20, cv_folds=5):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    train_metrics_list = []
    val_metrics_list = []

    for train_idx, val_idx in skf.split(df, df[status_var]):
        
        # Split original data frame
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        # Person-period transformation
        train_data = generate_person_time_data(
            train_df,
            time_var=time_var,
            status_var=status_var,
            bins=bins,
            cens="same"
        )

        train_data.drop(columns=['IDInternal', 'timeInternal', 'eventInternal', 'delta.lower','delta.upper', 'interval'], inplace=True)

        # Split into features and labels
        X_train_person_period = train_data.drop(columns=['N.delta'])
        y_train_person_period = train_data['N.delta'].astype(int)  # convert to int

        X_train = train_df.drop(columns=['survTime', 'statusComp'])
        y_train = train_df['statusComp'].astype(int)  # convert to int

        X_val = val_df.drop(columns=['survTime', 'statusComp'])
        y_val = val_df['statusComp'].astype(int)

        # Fit the neural net
        model = fit_nnet(X_train_person_period, y_train_person_period, hidden_layers=hidden_layers, alpha=alpha)

        # Predictions
        y_train_prob = model.predict_proba(X_train)[:, 1]
        y_val_prob = model.predict_proba(X_val)[:, 1]

        # Evaluate metrics
        train_brier, train_auc, train_pr_auc = evaluate_metrics(y_train, y_train_prob)
        val_brier, val_auc, val_pr_auc = evaluate_metrics(y_val, y_val_prob)

        train_metrics_list.append({
            "Brier": train_brier,
            "ROC AUC": train_auc,
            "PR-AUC": train_pr_auc
        })
        val_metrics_list.append({
            "Brier": val_brier,
            "ROC AUC": val_auc,
            "PR-AUC": val_pr_auc
        })

    # Average metrics across folds
    avg_train_metrics = {
        k: np.nanmean([m[k] for m in train_metrics_list])
        for k in train_metrics_list[0]
    }
    avg_val_metrics = {
        k: np.nanmean([m[k] for m in val_metrics_list])
        for k in val_metrics_list[0]
    }

    return {
        "Train": avg_train_metrics,
        "Validation": avg_val_metrics
    }

def objective_nn(params):
    hidden_layers, alpha, bins = params
    result = nnet_cv(df, time_var="survTime", status_var="statusComp", hidden_layers=hidden_layers, alpha=alpha, bins=bins)
    return -result['Validation']['PR-AUC']

def print_formatted_metrics(result):
    print("📊 Train Metrics:")
    print(f"  Mean Brier: {result['Train']['Brier']:.4f}")
    print(f"  Mean ROC-AUC: {result['Train']['ROC AUC']:.4f}")
    print(f"  Mean PR-AUC: {result['Train']['PR-AUC']:.4f}\n")

    print("📊 Validation Metrics:")
    print(f"  Mean Brier: {result['Validation']['Brier']:.4f}")
    print(f"  Mean ROC-AUC: {result['Validation']['ROC AUC']:.4f}")
    print(f"  Mean PR-AUC: {result['Validation']['PR-AUC']:.4f}")

def run_nnet(df):

    res = gp_minimize(objective_nn,
                    [Integer(1, 10), Real(0.01, 0.5), Integer(5, 50, name='bins')],
                    n_calls=20,
                    random_state=0)

    best_size = res.x[0]
    best_alpha = res.x[1]
    best_bins = res.x[2]

    print(f"Best size: {best_size}, Best alpha: {best_alpha}, Best bins: {best_bins}")

    result_nnet = nnet_cv(df, time_var="survTime", status_var="statusComp", hidden_layers=best_size, alpha=best_alpha, bins=best_bins)

    print_formatted_metrics(result_nnet)


#####  CoxPH Functions #####


def fit_coxph_pipeline(df, duration_col, event_col, time_horizons):
    """
    Fit CoxPH and compute C-index, Mean Brier score, Mean ROC AUC, and Mean PR-AUC.
    """
    X = df.drop([duration_col, event_col], axis=1)
    y_time = df[duration_col].values
    y_event = df[event_col].values

    # Fit CoxPH model
    cph = CoxPHFitter(penalizer=0.01)  # add small penalizer for stability
    cph.fit(df, duration_col=duration_col, event_col=event_col)

    # Predict risk scores (partial hazards)
    risk_scores = -cph.predict_partial_hazard(X).values.flatten()

    # 📈 Concordance index
    c_index = concordance_index(y_time, risk_scores, y_event)
    metrics = {"C-index": c_index}

    # 📉 Brier score, ROC AUC, and PR-AUC at each time horizon
    surv_funcs = cph.predict_survival_function(X)
    for t in time_horizons:
        if t in surv_funcs.index:
            surv_probs = surv_funcs.loc[t].values
        else:
            # Interpolate for time horizons not directly in surv_funcs
            surv_probs = surv_funcs.apply(lambda s: np.interp(t, surv_funcs.index, s), axis=0).values

        pred_risk = 1 - surv_probs
        observed = (y_time <= t).astype(int)

        # Brier score
        brier = brier_score_loss(observed, pred_risk)
        metrics[f"Brier_{t}"] = brier

        # ROC AUC and PR-AUC (only if both classes present)
        if len(np.unique(observed)) > 1:
            auc = roc_auc_score(observed, pred_risk)
            pr_auc = average_precision_score(observed, pred_risk)
        else:
            auc = np.nan  # Cannot compute AUC with 1 class
            pr_auc = np.nan  # Cannot compute PR-AUC with 1 class
        metrics[f"AUC_{t}"] = auc
        metrics[f"PR_AUC_{t}"] = pr_auc

    return metrics, cph


def fit_coxph_pipeline_cv(df, duration_col, event_col, cv_folds=5):
    
    """
    Perform k-fold CV for CoxPH and return averaged metrics.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    train_metrics_list = []
    val_metrics_list = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df[event_col])):
        
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)

        time_horizons_train = np.sort(df_train['survTime'].unique())
        time_horizons_val = np.sort(df_val['survTime'].unique())

        # Fit model and compute metrics
        train_metrics, _ = fit_coxph_pipeline(
            df=df_train,
            duration_col=duration_col,
            event_col=event_col,
            time_horizons=time_horizons_train
        )
        
        val_metrics, _ = fit_coxph_pipeline(
            df=df_val,
            duration_col=duration_col,
            event_col=event_col,
            time_horizons=time_horizons_val
        )

        train_metrics_list.append(train_metrics)
        val_metrics_list.append(val_metrics)

    # Collect all keys from all folds
    all_keys = set()
    for m in train_metrics_list + val_metrics_list:
        all_keys.update(m.keys())

    # Average metrics safely
    def average_metrics(metrics_list):
        avg_metrics = {}
        for key in all_keys:
            values = [m[key] for m in metrics_list if key in m and not np.isnan(m[key])]
            avg_metrics[key] = np.mean(values) if values else np.nan
        return avg_metrics

    avg_train_metrics = average_metrics(train_metrics_list)
    avg_val_metrics = average_metrics(val_metrics_list)

    return {
        "Train": avg_train_metrics,
        "Validation": avg_val_metrics
    }


def summarize_cv_metrics(metrics_dict):
    """Compute mean C-index, Brier, ROC-AUC, and PR-AUC from cv_metrics."""
    c_index = metrics_dict['C-index']

    # Extract and average Brier scores
    brier_values = [v for k, v in metrics_dict.items() if k.startswith('Brier_') and not np.isnan(v)]
    mean_brier = np.mean(brier_values) if brier_values else np.nan

    # Extract and average ROC-AUC
    auc_values = [v for k, v in metrics_dict.items() if k.startswith('AUC_') and not np.isnan(v)]
    mean_auc = np.mean(auc_values) if auc_values else np.nan

    # Extract and average PR-AUC
    pr_auc_values = [v for k, v in metrics_dict.items() if k.startswith('PR_AUC_') and not np.isnan(v)]
    mean_pr_auc = np.mean(pr_auc_values) if pr_auc_values else np.nan

    return {
        "C-index": c_index,
        "Mean Brier": mean_brier,
        "Mean ROC-AUC": mean_auc,
        "Mean PR-AUC": mean_pr_auc
    }

def summarize_cv_metrics(metrics_dict, metrics_type):
    """Compute mean C-index, Brier, ROC-AUC, and PR-AUC from cv_metrics."""
    c_index = metrics_dict['C-index']

    # Extract and average Brier scores
    brier_values = [v for k, v in metrics_dict.items() if k.startswith('Brier_') and not np.isnan(v)]
    mean_brier = np.mean(brier_values) if brier_values else np.nan

    # Extract and average ROC-AUC
    auc_values = [v for k, v in metrics_dict.items() if k.startswith('AUC_') and not np.isnan(v)]
    mean_auc = np.mean(auc_values) if auc_values else np.nan

    # Extract and average PR-AUC
    pr_auc_values = [v for k, v in metrics_dict.items() if k.startswith('PR_AUC_') and not np.isnan(v)]
    mean_pr_auc = np.mean(pr_auc_values) if pr_auc_values else np.nan

    results = {
        "C-index": c_index,
        "Mean Brier": mean_brier,
        "Mean ROC-AUC": mean_auc,
        "Mean PR-AUC": mean_pr_auc
    }

    # Print results
    print(f"📊 {metrics_type} Metrics:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

def run_coxph(df):

    # Run 5-fold CV
    cv_metrics = fit_coxph_pipeline_cv(
        df,
        duration_col='survTime',    # time-to-event column
        event_col='statusComp'     # event indicator column
    )

    summarize_cv_metrics(cv_metrics['Train'], "Train")
    summarize_cv_metrics(cv_metrics['Validation'], "Validation")
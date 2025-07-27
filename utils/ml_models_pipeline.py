from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

from skopt import gp_minimize
from skopt.space import Integer, Real

from lifelines.exceptions import ConvergenceError

from utils.person_period import generate_person_time_data


### ===== Shared CV & Evaluation Functions ===== ###

def evaluate_metrics(y_true, y_prob):
    """Compute Brier, ROC AUC, and PR-AUC for given predictions."""
    brier = brier_score_loss(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    return brier, auc, pr_auc

def model_cv(df, model_fn, param_dict, cv_folds=5, time_var="survTime", status_var="statusComp", bins=20):

    """
    Generic CV function for any classifier model.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    train_metrics_list, val_metrics_list = [], []

    for train_idx, val_idx in skf.split(df, df[status_var]):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

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

        X_train = train_df.drop(columns=[time_var, status_var])
        y_train = train_df[status_var].astype(int)

        X_val = val_df.drop(columns=[time_var, status_var])
        y_val = val_df[status_var].astype(int)

        # Clean column names to remove problematic characters
        X_train_person_period.columns = X_train_person_period.columns.str.replace(r"[\[\]<>]", "_", regex=True)
        X_train.columns = X_train.columns.str.replace(r"[\[\]<>]", "_", regex=True)
        X_val.columns = X_val.columns.str.replace(r"[\[\]<>]", "_", regex=True)

        model = model_fn(**param_dict)
        model.fit(X_train_person_period, y_train_person_period)

        y_train_prob = model.predict_proba(X_train)[:, 1]
        y_val_prob = model.predict_proba(X_val)[:, 1]

        train_brier, train_auc, train_pr_auc = evaluate_metrics(y_train, y_train_prob)
        val_brier, val_auc, val_pr_auc = evaluate_metrics(y_val, y_val_prob)

        train_metrics_list.append({
            "Mean Brier": train_brier,
            "Mean ROC-AUC": train_auc,
            "Mean PR-AUC": train_pr_auc
        })
        val_metrics_list.append({
            "Mean Brier": val_brier,
            "Mean ROC-AUC": val_auc,
            "Mean PR-AUC": val_pr_auc
        })

    avg_train_metrics = {k: np.nanmean([m[k] for m in train_metrics_list]) for k in train_metrics_list[0]}
    avg_val_metrics = {k: np.nanmean([m[k] for m in val_metrics_list]) for k in val_metrics_list[0]}

    return {"Train": avg_train_metrics, "Val": avg_val_metrics}


### ===== Logistic Regression ===== ###

def run_logreg(df, time_var="survTime", status_var="statusComp"):
    def objective(params):
        C, bins = params
        result = model_cv(df, LogisticRegression, {"C": C, "solver": "liblinear"}, cv_folds=5, time_var=time_var, status_var=status_var, bins=bins)
        return -result['Val']['Mean PR-AUC']

    res = gp_minimize(objective, [Real(0.001, 10, prior='log-uniform'), Integer(5, 50, name='bins')], n_calls=15, random_state=0)
    best_C = res.x[0]
    best_bins = res.x[1]

    return model_cv(df, LogisticRegression, {"C": best_C, "solver": "liblinear"}, cv_folds=5, time_var=time_var, status_var=status_var, bins=best_bins)


### ===== XGBoost ===== ###

def run_xgboost(df, time_var="survTime", status_var="statusComp"):

    def objective(params):

        max_depth, learning_rate, bins = params

        result = model_cv(df, XGBClassifier, {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "use_label_encoder": False,
            "eval_metric": "logloss"
        }, cv_folds=5,
           time_var=time_var,
           status_var=status_var,
           bins=bins)

        return -result['Val']['Mean PR-AUC']

    res = gp_minimize(objective,
                      [Integer(3, 10), Real(0.01, 0.3), Integer(5, 50, name='bins')],
                      n_calls=15, random_state=0)
    best_max_depth, best_lr, best_bins = res.x

    return model_cv(df, XGBClassifier, {
        "max_depth": best_max_depth,
        "learning_rate": best_lr,
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }, cv_folds=5, time_var=time_var, status_var=status_var, bins=best_bins)


### ===== SVM (with probability calibration) ===== ###

def run_svm(df, time_var="survTime", status_var="statusComp"):
    def objective(params):
        C, bins = params
        result = model_cv(df, SVC, {"C": C, "kernel": "rbf", "probability": True}, cv_folds=5,
                           time_var=time_var,
                           status_var=status_var,
                           bins=bins)
        return -result['Val']['Mean PR-AUC']

    res = gp_minimize(objective, [Real(0.001, 10, prior='log-uniform'), Integer(5, 50, name='bins')], n_calls=15, random_state=0)
    best_C, best_bins = res.x

    return model_cv(df, SVC, {"C": best_C, "kernel": "rbf", "probability": True}, cv_folds=5, time_var=time_var, status_var=status_var, bins=best_bins)


### ===== Random Forest  ===== ###

def run_random_forest(df, time_var="survTime", status_var="statusComp", cv_folds=5):
    """
    Wrapper for Random Forest with Bayesian hyperparameter tuning.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset with survival time and status.
    time_var : str
        Name of survival time column.
    status_var : str
        Name of event/censoring column.
    cv_folds : int
        Number of cross-Val folds.

    Returns
    -------
    dict
        Averaged CV metrics for train and Val sets.
    """
    
    # Objective function for Bayesian Optimization
    def objective_rf(params):
        n_estimators, max_depth, min_samples_split, bins = params

        param_dict = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "random_state": 42,
            "n_jobs": -1  # use all cores
        }

        result = model_cv(
            df,
            model_fn=RandomForestClassifier,
            param_dict=param_dict,
            cv_folds=cv_folds,
            time_var=time_var,
            status_var=status_var,
            bins=bins
        )

        # Minimize negative Mean PR-AUC on Val set
        return -result['Val']['Mean PR-AUC']

    # Search space for hyperparameters
    search_space = [
        Integer(50, 300, name='n_estimators'),          # Number of trees
        Integer(3, 20, name='max_depth'),               # Max tree depth
        Integer(2, 20, name='min_samples_split'),       # Min samples to split
        Integer(5, 50, name='bins')                     # Discretization bins
    ]

    # Run Bayesian optimization
    res = gp_minimize(
        objective_rf,
        search_space,
        n_calls=30,
        random_state=0
    )

    # Extract best hyperparameters
    best_params = {
        "n_estimators": res.x[0],
        "max_depth": res.x[1],
        "min_samples_split": res.x[2],
        "random_state": 42,
        "n_jobs": -1
    }
    best_bins = res.x[3]

    #print(f"✅ Best RF hyperparameters: {best_params}, bins={best_bins}")

    # Run final CV with best hyperparameters
    rf_results = model_cv(
        df,
        model_fn=RandomForestClassifier,
        param_dict=best_params,
        cv_folds=cv_folds,
        time_var=time_var,
        status_var=status_var,
        bins=best_bins
    )

    return rf_results

### ===== Neural Nets  ===== ###

def run_mlpclassifier(df, time_var="survTime", status_var="statusComp"):
    """
    Tune and evaluate MLPClassifier with multiple hidden layers using model_cv.
    """

    def objective_mlp(params):
        n_layers, size1, size2, size3, alpha, bins = params

        # Build hidden layer sizes tuple based on number of layers
        hidden_layers = tuple([size1, size2, size3][:n_layers])

        param_dict = {
            "hidden_layer_sizes": hidden_layers,  # variable number of layers
            "alpha": alpha,
            "activation": "relu",  # could also tune activation if desired
            "max_iter": 500,
            "solver": "adam",
            "random_state": 42
        }

        result = model_cv(
            df,
            model_fn=MLPClassifier,
            param_dict=param_dict,
            time_var=time_var,
            status_var=status_var,
            bins=bins
        )

        return -result['Val']['Mean PR-AUC']  # Negative for minimization

    # Search space:
    search_space = [
        Integer(1, 3, name="n_layers"),        # Number of hidden layers
        Integer(1, 64, name="size1"),        # Size of layer 1
        Integer(1, 64, name="size2"),        # Size of layer 2
        Integer(1, 64, name="size3"),        # Size of layer 3
        Real(0.001, 1, name="alpha"),       # L2 regularization strength
        Integer(5, 50, name="bins")            # Number of bins for person-period
    ]

    # Run Bayesian optimization
    res = gp_minimize(
        objective_mlp,
        search_space,
        n_calls=30,  # increase if needed
        random_state=0
    )

    # Extract best hyperparameters
    best_n_layers = res.x[0]
    best_sizes = tuple([res.x[1], res.x[2], res.x[3]][:best_n_layers])
    best_alpha = res.x[4]
    best_bins = res.x[5]

    #print(f"Best architecture: {best_sizes}, Best alpha: {best_alpha}, Best bins: {best_bins}")

    # Final model evaluation with best hyperparameters
    best_params = {
        "hidden_layer_sizes": best_sizes,
        "alpha": best_alpha,
        "activation": "relu",
        "max_iter": 500,
        "solver": "adam",
        "random_state": 42
    }

    result_mlp = model_cv(
        df,
        model_fn=MLPClassifier,
        param_dict=best_params,
        time_var=time_var,
        status_var=status_var,
        bins=best_bins
    )

    return result_mlp


##### Survival Models #####


from lifelines.exceptions import ConvergenceError

def run_coxph(df, duration_col="survTime", event_col="statusComp", cv_folds=5):
    """
    Wrapper for Cox Proportional Hazards model with k-fold cross-validation.
    Returns NaN metrics for folds where convergence fails.
    """
    def fit_and_eval(train_df, val_df, fitted_model=None):
        def evaluate(df_eval):
            try:
                X_eval = df_eval.drop([duration_col, event_col], axis=1)
                y_eval_time = df_eval[duration_col].values
                y_eval_event = df_eval[event_col].values

                risk_scores = -fitted_model.predict_partial_hazard(X_eval).values.flatten()
                c_index = concordance_index(y_eval_time, risk_scores, y_eval_event)
                metrics = {"C-index": c_index}

                time_horizons = np.sort(df_eval[duration_col].unique())
                surv_funcs = fitted_model.predict_survival_function(X_eval)

                for t in time_horizons:
                    surv_probs = (
                        surv_funcs.loc[t].values if t in surv_funcs.index
                        else surv_funcs.apply(lambda s: np.interp(t, surv_funcs.index, s), axis=0).values
                    )
                    pred_risk = 1 - surv_probs
                    observed = (y_eval_time <= t).astype(int)

                    metrics[f"Brier_{t}"] = brier_score_loss(observed, pred_risk)
                    if len(np.unique(observed)) > 1:
                        metrics[f"AUC_{t}"] = roc_auc_score(observed, pred_risk)
                        metrics[f"PR_AUC_{t}"] = average_precision_score(observed, pred_risk)
                    else:
                        metrics[f"AUC_{t}"] = np.nan
                        metrics[f"PR_AUC_{t}"] = np.nan

                return metrics
            except Exception as e:
                print(f"⚠️ Evaluation error: {e}. Returning NaNs.")
                return {"C-index": np.nan, "Mean Brier": np.nan, "Mean ROC-AUC": np.nan, "Mean PR-AUC": np.nan}

        return evaluate(train_df), evaluate(val_df)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    train_metrics, val_metrics = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df[event_col])):
        train_df, val_df = df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)
        cph_model = CoxPHFitter(penalizer=0.01)

        try:
            cph_model.fit(train_df, duration_col=duration_col, event_col=event_col)
            train_result, val_result = fit_and_eval(train_df, val_df, fitted_model=cph_model)
        except ConvergenceError as e:
            print(f"⚠️ Fold {fold+1}: Convergence error - {e}. Using NaNs for this fold.")
            nan_metrics = {"C-index": np.nan, "Mean Brier": np.nan, "Mean ROC-AUC": np.nan, "Mean PR-AUC": np.nan}
            train_result, val_result = nan_metrics, nan_metrics

        train_metrics.append(train_result)
        val_metrics.append(val_result)

    def avg_metrics(metrics_list):
        all_keys = {k for m in metrics_list for k in m}
        avg = {}
        for key in all_keys:
            values = [m[key] for m in metrics_list if key in m and not np.isnan(m[key])]
            mean_val = np.mean(values) if values else np.nan

            # Collapse time-based metrics into mean
            if key.startswith("Brier_"):
                avg["Mean Brier"] = avg.get("Mean Brier", []) + [mean_val]
            elif key.startswith("AUC_"):
                avg["Mean ROC-AUC"] = avg.get("Mean ROC-AUC", []) + [mean_val]
            elif key.startswith("PR_AUC_"):
                avg["Mean PR-AUC"] = avg.get("Mean PR-AUC", []) + [mean_val]
            else:
                avg[key] = mean_val

        # Final averaging for collapsed metrics
        for k in ["Mean Brier", "Mean ROC-AUC", "Mean PR-AUC"]:
            if k in avg:
                avg[k] = np.nanmean(avg[k])

        return avg

    return {
        "Train": avg_metrics(train_metrics),
        "Val": avg_metrics(val_metrics)
    }


def run_rsf(df, duration_col="survTime", event_col="statusComp", cv_folds=5):
    def to_structured_array(data):
        return np.array([(bool(e), t) for e, t in zip(data[event_col], data[duration_col])],
                        dtype=[('event', 'bool'), ('time', 'float')])

    def evaluate(rsf, X, y, raw_df):
        min_time, max_time = rsf.unique_times_[0], rsf.unique_times_[-1]
        candidate_times = np.sort(raw_df[duration_col].unique())
        time_horizons = [t for t in candidate_times if min_time <= t <= max_time]

        metrics = {}
        surv_funcs = rsf.predict_survival_function(X)

        for t in time_horizons:
            surv_probs = np.array([fn(t) for fn in surv_funcs])
            pred_risk = 1 - surv_probs
            observed = (raw_df[duration_col] <= t).astype(int)

            metrics[f"Brier_{t}"] = brier_score_loss(observed, pred_risk)
            if len(np.unique(observed)) > 1:
                metrics[f"AUC_{t}"] = roc_auc_score(observed, pred_risk)
                metrics[f"PR_AUC_{t}"] = average_precision_score(observed, pred_risk)
            else:
                metrics[f"AUC_{t}"] = np.nan
                metrics[f"PR_AUC_{t}"] = np.nan

        c_index = concordance_index_censored(y["event"], y["time"], rsf.predict(X))
        metrics["C-index"] = c_index[0]

        return metrics

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    train_metrics, val_metrics = [], []

    for train_idx, val_idx in skf.split(df, df[event_col]):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        X_train = train_df.drop([duration_col, event_col], axis=1)
        y_train = to_structured_array(train_df)

        X_val = val_df.drop([duration_col, event_col], axis=1)
        y_val = to_structured_array(val_df)

        rsf = RandomSurvivalForest(
            n_estimators=200,
            min_samples_split=10,
            min_samples_leaf=15,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42
        )
        rsf.fit(X_train, y_train)

        train_metrics.append(evaluate(rsf, X_train, y_train, train_df))
        val_metrics.append(evaluate(rsf, X_val, y_val, val_df))

    def avg_metrics(metrics_list):
        all_keys = {k for m in metrics_list for k in m}
        avg = {}
        for key in all_keys:
            values = [m[key] for m in metrics_list if key in m and not np.isnan(m[key])]
            mean_val = np.mean(values) if values else np.nan

            if key.startswith("Brier_"):
                avg["Mean Brier"] = avg.get("Mean Brier", []) + [mean_val]
            elif key.startswith("AUC_"):
                avg["Mean ROC-AUC"] = avg.get("Mean ROC-AUC", []) + [mean_val]
            elif key.startswith("PR_AUC_"):
                avg["Mean PR-AUC"] = avg.get("Mean PR-AUC", []) + [mean_val]
            else:
                avg[key] = mean_val

        for k in ["Mean Brier", "Mean ROC-AUC", "Mean PR-AUC"]:
            if k in avg:
                avg[k] = np.nanmean(avg[k])

        return avg

    return {
        "Train": avg_metrics(train_metrics),
        "Val": avg_metrics(val_metrics)
    }

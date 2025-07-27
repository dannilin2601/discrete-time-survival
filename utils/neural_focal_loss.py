import numpy as np
from sklearn.model_selection import StratifiedKFold
from focal_loss import BinaryFocalLoss
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf

from utils.person_period import generate_person_time_data
from utils.ml_models_pipeline import evaluate_metrics


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def model_cv_keras_focal(df, model_fn, param_dict, cv_folds=5,
                         time_var="survTime", status_var="statusComp", bins=20):

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

        train_data.drop(columns=['IDInternal', 'timeInternal', 'eventInternal',
                                 'delta.lower', 'delta.upper', 'interval'], inplace=True)

        # Features and labels
        X_train_person_period = train_data.drop(columns=['N.delta'])
        y_train_person_period = train_data['N.delta'].astype(int)

        X_train = train_df.drop(columns=[time_var, status_var])
        y_train = train_df[status_var].astype(int)

        X_val = val_df.drop(columns=[time_var, status_var])
        y_val = val_df[status_var].astype(int)

        # Clean column names
        X_train_person_period.columns = X_train_person_period.columns.str.replace(r"[\[\]<>]", "_", regex=True)
        X_train.columns = X_train.columns.str.replace(r"[\[\]<>]", "_", regex=True)
        X_val.columns = X_val.columns.str.replace(r"[\[\]<>]", "_", regex=True)

        # Build Keras model
        model = model_fn(input_dim=X_train_person_period.shape[1], **param_dict)

        with tf.device('/GPU:0'):

            model.fit(X_train_person_period, y_train_person_period,
                    epochs=100,
                    batch_size=2*2048,
                    verbose=0,
                    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

        # Predict on original input space (non-person-period)
        y_train_prob = model.predict(X_train, verbose=0).flatten()
        y_val_prob = model.predict(X_val, verbose=0).flatten()

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


from skopt import gp_minimize
from skopt.space import Integer, Real

def run_nn_focal(df, time_var="survTime", status_var="statusComp"):

    def build_nn(input_dim, hidden_sizes=(64,), gamma=2.0, lr=1e-3):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(input_dim,)))
        for size in hidden_sizes:
            model.add(tf.keras.layers.Dense(size, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=BinaryFocalLoss(gamma=gamma),
            metrics=["AUC"]
        )

        return model

    def objective(params):
        n_layers, size1, size2, size3, gamma, lr, bins = params
        hidden_layers = tuple([size1, size2, size3][:n_layers])

        param_dict = {
            "hidden_sizes": hidden_layers,
            "gamma": gamma,
            "lr": lr
        }

        def model_fn(input_dim, **kwargs):
            return build_nn(input_dim, **kwargs)

        result = model_cv_keras_focal(
            df,
            model_fn=model_fn,
            param_dict=param_dict,
            time_var=time_var,
            status_var=status_var,
            bins=bins
        )
        return -result['Val']['Mean PR-AUC']

    # Define the search space
    search_space = [
        Integer(1, 3, name="n_layers"),
        Integer(1, 128, name="size1"),
        Integer(1, 128, name="size2"),
        Integer(1, 128, name="size3"),
        Real(1.0, 5.0, name="gamma"),
        Real(1e-4, 1e-2, prior='log-uniform', name="lr"),
        Integer(5, 50, name="bins")
    ]

    res = gp_minimize(
        objective,
        search_space,
        n_calls=30,
        random_state=0
    )

    # Best hyperparameters
    best_n_layers = res.x[0]
    best_sizes = tuple([res.x[1], res.x[2], res.x[3]][:best_n_layers])
    best_gamma = res.x[4]
    best_lr = res.x[5]
    best_bins = res.x[6]

    best_param_dict = {
        "hidden_sizes": best_sizes,
        "gamma": best_gamma,
        "lr": best_lr
    }

    def model_fn_final(input_dim, **kwargs):
        return build_nn(input_dim, **kwargs)

    result_final = model_cv_keras_focal(
        df,
        model_fn=model_fn_final,
        param_dict=best_param_dict,
        time_var=time_var,
        status_var=status_var,
        bins=best_bins
    )

    return result_final


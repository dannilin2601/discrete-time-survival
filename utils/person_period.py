import numpy as np
import pandas as pd

def create_discrete_intervals(time, event, bins):
    """
    Replicates createDiscreteIntervals from https://github.com/ksuresh17/autoSurv.
    """
    unique_event_times = np.unique(time[event == 1])
    n_delta = min(bins, len(unique_event_times))
    probs_delta = np.linspace(0, 1, n_delta + 1)
    delta_upper = np.quantile(time[event == 1], probs_delta)
    
    return delta_upper[1:]  # drop first interval (t0=0 assumed)


def create_discrete_dat(time_internal, event_internal, data_x, delta_upper, cens="same"):
    """
    Replicates createDiscreteDat from https://github.com/ksuresh17/autoSurv.
    """
    delta_lower = np.concatenate(([0], delta_upper[:-1]))
    n_delta = len(delta_upper)
    id_internal = np.arange(1, len(data_x) + 1)
    dat_i = pd.concat([pd.Series(id_internal, name="IDInternal"),
                       pd.Series(time_internal, name="timeInternal"),
                       pd.Series(event_internal, name="eventInternal"),
                       data_x.reset_index(drop=True)], axis=1)

    # Repeat rows for each interval
    long_dat = pd.DataFrame(np.repeat(dat_i.values, n_delta, axis=0), columns=dat_i.columns)
    long_dat["delta.lower"] = np.tile(delta_lower, len(dat_i))
    long_dat["delta.upper"] = np.tile(delta_upper, len(dat_i))
    long_dat["interval"] = np.tile(np.arange(1, n_delta + 1), len(dat_i))
    long_dat["N.delta"] = np.nan

    # Treatment of censored observations
    if cens == "same":
        condition1 = long_dat["timeInternal"] > long_dat["delta.upper"]
        condition2 = (long_dat["eventInternal"] == 1) & (long_dat["timeInternal"] <= long_dat["delta.lower"])
        condition3 = (long_dat["eventInternal"] == 1)
        condition4 = long_dat["timeInternal"] >= 0.5 * (long_dat["delta.upper"] + long_dat["delta.lower"])

        long_dat.loc[condition1, "N.delta"] = 0
        long_dat.loc[~condition1 & condition3 & ~condition2, "N.delta"] = 1
        long_dat.loc[~condition1 & ~condition3 & (long_dat["timeInternal"] > long_dat["delta.lower"]), "N.delta"] = 0

    elif cens == "prev":
        long_dat.loc[long_dat["timeInternal"] > long_dat["delta.upper"], "N.delta"] = 0
        long_dat.loc[(long_dat["eventInternal"] == 1) & (long_dat["timeInternal"] > long_dat["delta.lower"]), "N.delta"] = 1

    elif cens == "half":
        condition1 = long_dat["timeInternal"] > long_dat["delta.upper"]
        condition2 = (long_dat["eventInternal"] == 1) & (long_dat["timeInternal"] <= long_dat["delta.lower"])
        condition3 = (long_dat["eventInternal"] == 1)
        condition4 = long_dat["timeInternal"] >= 0.5 * (long_dat["delta.upper"] + long_dat["delta.lower"])

        long_dat.loc[condition1, "N.delta"] = 0
        long_dat.loc[~condition1 & condition3 & ~condition2, "N.delta"] = 1
        long_dat.loc[~condition1 & ~condition3 & condition4, "N.delta"] = 0

    # Handle last interval events
    m = delta_upper[-1]
    mask_last_interval = (long_dat["timeInternal"] == m) & (long_dat["delta.upper"] == m)
    long_dat.loc[mask_last_interval, "N.delta"] = long_dat.loc[mask_last_interval, "N.delta"].fillna(0)

    # Drop rows where N.delta is NA
    long_dat = long_dat.dropna(subset=["N.delta"])
    long_dat["interval"] = long_dat["interval"].astype(int).astype("category")
    long_dat["N.delta"] = long_dat["N.delta"].astype(int).astype("category")

    return long_dat.reset_index(drop=True)


def gen_pt_data(dat, time_var, status_var, bins=5, w=None, cens="same"):
    """
    Replicates genPTdat from R.
    """
    dat_i = dat.copy()
    time_idx = dat.columns.get_loc(time_var)
    status_idx = dat.columns.get_loc(status_var)

    # Ignore events after prediction window
    if w is not None:
        dat_i.loc[dat_i[time_var] > w, status_var] = 0
        dat_i.loc[dat_i[time_var] > w, time_var] = w

    # Identify discrete intervals
    delta_upper = create_discrete_intervals(dat_i[time_var], dat_i[status_var], bins)

    # Replace last interval with w
    if w is not None:
        delta_upper[-1] = w

    # Create long format data
    dat_i_x = create_discrete_dat(dat_i[time_var].values,
                                  dat_i[status_var].values,
                                  dat_i.drop(columns=[time_var, status_var]),
                                  delta_upper,
                                  cens=cens)
    return dat_i_x

def generate_person_time_data(train_df, time_var="survTime", status_var="statusComp", bins=20, cens="same"):
    """
    Generates a person-time dataset from survival data.

    Parameters
    ----------
    train_df : pd.DataFrame
        The input dataframe with survival data.
    time_var : str
        Name of the time-to-event column.
    status_var : str
        Name of the event indicator column.
    bins : int
        Number of discrete time bins to create (default: 20).
    cens : str
        How to handle censored data: "same", "prev", or "half" (default: "same").

    Returns
    -------
    pd.DataFrame
        A person-period (long format) dataframe.
    """
    
    # Define max time (prediction window)
    w_pred = train_df[time_var].max()

    # Generate person-period dataset
    pt_data = gen_pt_data(
        dat=train_df,
        time_var=time_var,
        status_var=status_var,
        bins=bins,
        w=w_pred,
        cens=cens
    )

    return pt_data

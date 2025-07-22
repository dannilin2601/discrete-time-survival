import pandas as pd

def summarize_cv_metrics(model_results_list, dataset_name):
    """
    Combine metrics from multiple models into a single DataFrame.

    Parameters:
    - model_results_list: list of tuples (model_name, results_dict)
    - dataset_name: name of the dataset (used as index)

    Returns:
    - DataFrame with rows = metrics, columns = Model_Split
    """

    def flatten_results(results, model_name):
        """
        Flatten nested metrics into a dictionary:
        e.g., NeuralNet_Train_Brier -> value
        """
        flat_dict = {}
        for split, metrics in results.items():
            for metric, value in metrics.items():
                key = f"{model_name}_{split}_{metric}"
                flat_dict[key] = value
        return flat_dict

    combined_flat = {}

    # Flatten results for all models
    for model_name, results in model_results_list:
        combined_flat.update(flatten_results(results, model_name))

    # Convert to DataFrame
    df = pd.DataFrame(combined_flat, index=[dataset_name]).transpose().reset_index()
    df.columns = ["Metric_Split_Model", "Value"]

    # Split column into Model, Split, Metric
    df[['Model', 'Split', 'Metric']] = df['Metric_Split_Model'].str.split('_', expand=True)
    df = df.drop(columns=["Metric_Split_Model"])

    # Pivot table: rows = Metric, columns = Model_Split
    pivot_df = df.pivot_table(
        index="Metric",
        columns=["Model", "Split"],
        values="Value"
    )

    # Flatten multi-level columns
    pivot_df.columns = [f"{model}_{split}" for model, split in pivot_df.columns]
    pivot_df = pivot_df.reset_index()

    return pivot_df

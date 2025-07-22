
import pandas as pd


import pandas as pd

def summarize_cv_metrics(nn_results, coxph_results, dataset_name):
    def flatten_results(results, model_name):
        """
        Flatten nested metrics into a dictionary with keys like:
        NeuralNet_Train_Brier, NeuralNet_Validation_ROC AUC, etc.
        """
        flat_dict = {}
        for split, metrics in results.items():  # e.g., 'Train', 'Validation'
            for metric, value in metrics.items():
                key = f"{model_name}_{split}_{metric}"
                flat_dict[key] = value
        return flat_dict

    # Flatten both model results
    nn_flat = flatten_results(nn_results, "NeuralNet")
    coxph_flat = flatten_results(coxph_results, "CoxPH")

    # Combine results into one dictionary
    combined_flat = {**nn_flat, **coxph_flat}

    # Convert to DataFrame
    df = pd.DataFrame(combined_flat, index=[dataset_name])

    # Transpose for metrics as rows and models as columns
    summary_df = df.transpose().reset_index()
    summary_df.columns = ["Metric", dataset_name]

    # Split "Metric" column into Model, Split, and Metric for clarity
    summary_df[['Model', 'Split', 'Metric Name']] = summary_df['Metric'].str.split('_', n=2, expand=True)
    summary_df = summary_df.drop(columns=["Metric"])

    # Pivot: rows = Metric Name, columns = Model_Split
    final_df = summary_df.pivot_table(
        index="Metric Name",
        columns=["Model", "Split"],
        values=dataset_name
    )

    return final_df


import pandas as pd
import numpy as np

# modelop.init
def begin():
    """A function to declare model-specific variables used in ROI computation
    """
    global amount_field, label_field, score_field
    global baseline_metrics, cost_multipliers

    amount_field = "loan_amnt"  # Column containing transaction amount
    label_field = "loan_status"  # Column containing ground_truth
    score_field = "score"  # Column containing model prediction

    # Classification metrics on baseline data
    baseline_metrics = {"TPR": 0.4, "FPR": 0.1, "TNR": 0.2, "FNR": 0.3}

    # ROI cost multipliers for each classification case
    cost_multipliers = {
        "TP": 2,
        "FP": -1.5,
        "TN": 1.5,
        "FN": -2,
    }
    pass


# modelop.metrics
def metrics(df_sample):
    """Function to classify records & compute actual ROI given labeled & scored DataFrame.

    Args:
        df_sample (pd.DataFrame): Slice of Production data

    Yields:
        dict: Name of transaction field and actual ROI
    """

    # Classify each record in dataframe
    for idx in range(len(df_sample)):
        if df_sample.iloc[idx][label_field] == df_sample.iloc[idx][score_field]:
            df_sample["record_class"] = (
                "TP" if df_sample.iloc[idx][label_field] == 0 else "TN"
            )
        elif df_sample.iloc[idx][label_field] < df_sample.iloc[idx][score_field]:
            df_sample["record_class"] = "FN"
        else:
            df_sample["record_class"] = "FP"

    actual_ROI = compute_actual_ROI(df_sample)
    projected_ROI = None

    yield {
        "actual_ROI": actual_ROI,
        "projected_ROI": projected_ROI,
        "amount_field": amount_field,
        "ROI": [
            {
                "test_name": "ROI",
                "test_category": "ROI",
                "test_type": "ROI",
                "test_id": "ROI",
                "values": {
                    "actual_ROI": actual_ROI,
                    "projected_ROI": projected_ROI,
                    "amount_field": amount_field,
                    "baseline_metrics": baseline_metrics,
                    "cost_multipliers": cost_multipliers,
                },
            }
        ],
    }


def compute_actual_ROI(data):
    """Helper function to compute actual ROI.

    Args:
        data (pd.DataFrame): Input DataFrame containing record_class

    Returns:
        float: actual ROI
    """
    actual_ROI = 0
    for idx in range(len(data)):
        actual_ROI += (
            data.iloc[idx][amount_field]
            * cost_multipliers[data.iloc[idx]["record_class"]]
        )

    return actual_ROI

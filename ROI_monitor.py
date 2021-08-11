# modelop.init
def begin():
    """A function to declare model-specific variables used in ROI computation"""
    global amount_field, label_field, score_field
    global baseline_metrics, cost_multipliers

    amount_field = "credit_amount"  # Column containing transaction amount
    label_field = "label_value"  # Column containing ground_truth
    score_field = "score"  # Column containing model prediction

    # Classification metrics on baseline data
    baseline_metrics = {
        "TNR": 0.8772321428571429,
        "TPR": 0.53125,
    }
    # ROI cost multipliers for each classification case
    cost_multipliers = {
        "TP": 1.5,
        "FP": -2,
        "TN": 2,
        "FN": -1.5,
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
    # Positive Class Labeled as 1
    # Negative Class labeled as 0

    for idx in range(len(df_sample)):
        if df_sample.iloc[idx][label_field] == df_sample.iloc[idx][score_field]:
            df_sample["record_class"] = (
                "TP" if df_sample.iloc[idx][label_field] == 1 else "TN"
            )
        elif df_sample.iloc[idx][label_field] < df_sample.iloc[idx][score_field]:
            df_sample["record_class"] = "FP"
        else:
            df_sample["record_class"] = "FN"

    # Compute actual and projected ROIs
    actual_roi = compute_actual_roi(df_sample)
    projected_roi = compute_projected_roi(df_sample)

    yield {
        "actual_roi": actual_roi,
        "projected_roi": projected_roi,
        "amount_field": amount_field,
        "ROI": [
            {
                "test_name": "ROI",
                "test_category": "ROI",
                "test_type": "ROI",
                "test_id": "ROI",
                "values": {
                    "actual_roi": actual_roi,
                    "projected_roi": projected_roi,
                    "amount_field": amount_field,
                    "baseline_metrics": baseline_metrics,
                    "cost_multipliers": cost_multipliers,
                },
            }
        ],
    }


def compute_actual_roi(data):
    """Helper function to compute actual ROI.

    Args:
        data (pd.DataFrame): Input DataFrame containing record_class

    Returns:
        float: actual ROI
    """
    actual_roi = 0
    for idx in range(len(data)):
        actual_roi += (
            data.iloc[idx][amount_field]
            * cost_multipliers[data.iloc[idx]["record_class"]]
        )

    return round(actual_roi, 2)


def compute_projected_roi(data):
    """Helper function to compute projected ROI.

    Args:
        data (pd.DataFrame): Input DataFrame containing record_class

    Returns:
        float: projected ROI
    """
    projected_roi = 0
    for idx in range(len(data)):
        projected_roi += data.iloc[idx][amount_field] * (
            (data.iloc[idx][score_field] == 1)
            * (
                baseline_metrics["TPR"] * cost_multipliers["TP"]
                + (1 - baseline_metrics["TPR"] * cost_multipliers["FP"])
            )
            + (data.iloc[idx][score_field] == 0)
            * (
                baseline_metrics["TNR"] * cost_multipliers["TN"]
                + (1 - baseline_metrics["TNR"] * cost_multipliers["FN"])
            )
        )

    return round(projected_roi, 2)

import numpy as np
import polars as pl


def calculate_normalized_entropy(attributions: list[float]) -> float:
    """Calculate the normalized entropy of features attributions

    Args:
        attributions (list[float]): Feature attributions

    Returns:
        float: Normalized entropy
    """
    attributions_np = np.array(attributions)
    entropy = -np.sum(attributions_np * np.log2(attributions_np + 1e-11))
    normalized_entropy = entropy / np.log2(len(attributions))
    return normalized_entropy


def calculate_sparsity_metrics(
    df: pl.DataFrame, decision_boundary: float
) -> dict[str, float]:
    """Calculate sparsity metrics.

    Args:
        df (pl.DataFrame): Dataframe with attributions.
        decision_boundary (float): Decision boundary to predict explanations.

    Returns:
        dict[str, float]: Dictionary with sparsity metrics.
    """
    entropy = df.select(
        entropy=pl.col("attributions").map_elements(calculate_normalized_entropy)
    )["entropy"].mean()
    no_predictions = (df["attributions"].list.max() < decision_boundary).mean()
    attributions_sum_zero = (df["attributions"].list.sum() == 0).mean()
    return {
        "entropy": entropy,
        "no_predictions": no_predictions,
        "attributions_sum_zero": attributions_sum_zero,
    }

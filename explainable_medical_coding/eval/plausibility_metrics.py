# ruff: noqa: E402
from pathlib import Path

import sklearn.metrics
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

import numpy as np
import polars as pl
import sklearn
import torch
from datasets import DatasetDict
from rich.pretty import pprint
from transformers import AutoTokenizer

from explainable_medical_coding.config.factories import get_explainability_method
from explainable_medical_coding.eval.sparsity_metrics import calculate_sparsity_metrics
from explainable_medical_coding.utils.tokenizer import TargetTokenizer
from explainable_medical_coding.utils.analysis import get_explanations


def attributions2token_ids(
    attributions: list[float], decision_boundary: float
) -> list[int]:
    """Convert attributions, which is one score per token, to a list of token ids.

    Args:
        attributions (list[float]): Attribution scores.
        decision_boundary (float): Decision boundary to select which tokens to predict as explanations.

    Returns:
        list[int]: A list of token ids.
    """
    return np.where(np.array(attributions) > decision_boundary)[0].tolist()


def calculate_error_matrix_from_dataframe(
    df: pl.DataFrame, decision_boundary: float
) -> pl.DataFrame:
    """Calculate the true positives, false positives and false negatives for a given decision boundary.
    Add the statistics to the dataframe.

    Args:
        df (pl.DataFrame): A dataframe with attributions and evidence_token_ids.
        decision_boundary (float): Decision boundary to predict explanations.

    Returns:
        pl.DataFrame: A dataframe containing the true positives, false positives and false negatives for each code.
    """
    pred_groundtruth = df.select(
        predicted_token_ids=pl.col("attributions").map_elements(
            lambda x: attributions2token_ids(x, decision_boundary)
        ),
        evidence_token_ids=pl.col("evidence_token_ids"),
    )
    tp = pred_groundtruth.select(
        tp=pl.col("predicted_token_ids")
        .list.set_intersection("evidence_token_ids")
        .list.len()
    )
    fp = pred_groundtruth.select(
        fp=pl.col("predicted_token_ids")
        .list.set_difference("evidence_token_ids")
        .list.len()
    )
    fn = pred_groundtruth.select(
        fn=pl.col("evidence_token_ids")
        .list.set_difference("predicted_token_ids")
        .list.len()
    )
    return df.with_columns(tp=tp["tp"], fp=fp["fp"], fn=fn["fn"])


def calculate_evidence_prediction_metrics_from_dataframe(
    df: pl.DataFrame,
) -> tuple[float, float, float]:
    """Calculate the precision, recall and F1 score for a given decision boundary.

    Args:
        df (pl.DataFrame): A dataframe with attributions and one hot encoded evidence.
        decision_boundary (float): Decision boundary.

    Returns:
        tuple[float, float, float]: Precision, recall and F1 score.
    """
    tp = df["tp"].sum()
    fp = df["fp"].sum()
    fn = df["fn"].sum()
    precision = tp / (tp + fp + 1e-11)
    recall = tp / (tp + fn + 1e-11)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-11)
    return precision, recall, f1


def calculate_evidence_ranking_metrics_from_dataframe(
    df: pl.DataFrame, k: int = 100
) -> tuple[float, float]:
    """Calculate the precision@k and recall@k for a given K.

    Args:
        df (pl.DataFrame): A dataframe with attributions and one hot encoded evidence.
        k (int, optional): K. Defaults to 100.

    Returns:
        tuple[float, float]: Precision@k and recall@k.
    """
    df = df.with_columns(
        pl.col("attributions")
        .map_elements(lambda x: np.argsort(x)[-k:].tolist())
        .alias("top_k_indices")
    )
    number_of_relevant_tokens = df.select(
        pl.col("top_k_indices").list.set_intersection("evidence_token_ids").list.len()
    )["top_k_indices"].sum()
    number_of_possible_relevant_tokens = df.select(
        pl.col("evidence_token_ids").list.len()
    )["evidence_token_ids"].sum()
    number_of_recommended_tokens = len(df) * k
    recall = number_of_relevant_tokens / number_of_possible_relevant_tokens
    precision = number_of_relevant_tokens / number_of_recommended_tokens
    return precision, recall


def calculate_iou(df: pl.DataFrame) -> float:
    """Calculate the precision@k and recall@k for a given K.

    Args:
        df (pl.DataFrame): A dataframe with attributions and one hot encoded evidence.
        k (int, optional): K. Defaults to 100.

    Returns:
        tuple[float, float]: Precision@k and recall@k.
    """
    df = df.with_columns(
        pl.struct("attributions", "evidence_token_ids")
        .map_elements(
            lambda x: np.argsort(x["attributions"])[
                -len(x["evidence_token_ids"]) :
            ].tolist()
        )
        .alias("top_k_indices")
    )
    intersection = df.select(
        pl.col("top_k_indices").list.set_intersection("evidence_token_ids").list.len()
    )["top_k_indices"]
    union = df.select(
        pl.col("top_k_indices").list.set_union("evidence_token_ids").list.len()
    )["top_k_indices"]
    iou = (intersection / union).mean()
    return iou


def calculate_kl_divergence(
    attributions: list[float], evidence_token_ids: list[int]
) -> float:
    attributions_np = np.array(attributions)
    evidence_attributions = np.zeros(attributions_np.shape)
    evidence_attributions[evidence_token_ids] = 1 / len(evidence_token_ids)
    kl_divergence = np.sum(
        evidence_attributions
        * np.log((evidence_attributions + 1e-11) / (attributions_np + 1e-11))
    )
    return kl_divergence


def token_ids2token_id_sequences(token_ids: list[int]) -> list[set[int]]:
    """This function takes a list of token ids and returns a list of sets. each set contains the token ids that are sequential.

    Args:
        token_ids (list[int]): A list of token ids.

    Returns:
        list[set[int]]: A list of sets. Each set contains the token ids that are sequential.
    """
    token_id_sequences = []
    token_id_sequence: set[int] = set()
    for token_id in token_ids:
        if len(token_id_sequence) == 0:
            token_id_sequence.add(token_id)
        elif token_id == max(token_id_sequence) + 1:
            token_id_sequence.add(token_id)
        else:
            token_id_sequences.append(token_id_sequence)
            token_id_sequence = set()
            token_id_sequence.add(token_id)
    token_id_sequences.append(token_id_sequence)
    return token_id_sequences


def token_id_sequence_metrics(
    attributions: list[float], evidence_token_ids: list[int], decision_boundary: float
) -> tuple[float, float]:
    predicted_token_ids = set(attributions2token_ids(attributions, decision_boundary))
    evidence_token_id_sequences = token_ids2token_id_sequences(evidence_token_ids)

    number_of_evidence_sequences = len(evidence_token_id_sequences)
    number_of_predicted_evidence_sequences = 0
    number_of_prediced_tokens_in_evidence_sequences = 0
    number_of_tokens_in_evidence_sequences = 0

    for evidence_token_id_sequence in evidence_token_id_sequences:
        intersection = evidence_token_id_sequence.intersection(predicted_token_ids)
        if len(intersection) > 0:
            number_of_predicted_evidence_sequences += 1
            number_of_tokens_in_evidence_sequences += len(evidence_token_id_sequence)
            number_of_prediced_tokens_in_evidence_sequences += len(intersection)

    evidence_span_token_recall = number_of_prediced_tokens_in_evidence_sequences / (
        number_of_tokens_in_evidence_sequences + 1e-11
    )
    evidence_span_recall = number_of_predicted_evidence_sequences / (
        number_of_evidence_sequences + 1e-11
    )

    return evidence_span_token_recall, evidence_span_recall


def calculate_span_token_recall_and_evidence_span_token_recall_from_dataframe(
    df: pl.DataFrame, decision_boundary: float
) -> tuple[float, float]:
    """Calculate the span token recall for a given decision boundary.

    Args:
        df (pl.DataFrame): A dataframe with attributions and one hot encoded evidence.
        decision_boundary (float): Decision boundary.

    Returns:
        tuple[float, float]: Span token recall and evidence span token recall.
    """
    evidence_span_token_recall, evidence_span_recall = (
        df.select(
            pl.struct("attributions", "evidence_token_ids")
            .map_elements(
                lambda row: token_id_sequence_metrics(
                    row["attributions"],
                    row["evidence_token_ids"],
                    decision_boundary,
                )
            )
            .alias("temp")
        )
        .with_columns(
            pl.col("temp").list.to_struct(
                fields=["evidence_span_token_recall", "evidence_span_recall"]
            )
        )
        .unnest("temp")
        .mean()
    )
    return float(evidence_span_token_recall[0]), float(evidence_span_recall[0])


def calculate_auprc(df: pl.DataFrame) -> float:
    """Find the decision boundary that maximizes the F1 score.

    Args:
        df (pl.DataFrame): A dataframe with attributions and one hot encoded evidence.

    Returns:
        float: Area under the precision recall curve.
    """

    decision_boundaries = np.linspace(0, 1, 101)
    recall_scores = np.zeros_like(decision_boundaries)
    precision_scores = np.zeros_like(decision_boundaries)
    for idx, decision_boundary in enumerate(decision_boundaries):
        df_db = calculate_error_matrix_from_dataframe(df, decision_boundary)
        precision, recall, __ = calculate_evidence_prediction_metrics_from_dataframe(
            df_db
        )
        precision_scores[idx] = precision
        recall_scores[idx] = recall
    return sklearn.metrics.auc(recall_scores, precision_scores)


def calculate_plausibility_metrics(
    df: pl.DataFrame, decision_boundary: float
) -> dict[str, str | float]:
    df = calculate_error_matrix_from_dataframe(df, decision_boundary)
    precision, recall, f1 = calculate_evidence_prediction_metrics_from_dataframe(df)
    precision_1, recall_1 = calculate_evidence_ranking_metrics_from_dataframe(df, k=1)
    precision_5, recall_5 = calculate_evidence_ranking_metrics_from_dataframe(df, k=5)
    precision_10, recall_10 = calculate_evidence_ranking_metrics_from_dataframe(
        df, k=10
    )
    precision_50, recall_50 = calculate_evidence_ranking_metrics_from_dataframe(
        df, k=50
    )
    precision_100, recall_100 = calculate_evidence_ranking_metrics_from_dataframe(
        df, k=100
    )
    kl_divergence = df.select(
        kl_divergence=pl.struct("attributions", "evidence_token_ids").map_elements(
            lambda row: calculate_kl_divergence(
                row["attributions"], row["evidence_token_ids"]
            )
        )
    )["kl_divergence"].mean()

    average_number_of_predicted_tokens = (df["tp"] + df["fp"]).mean()

    (
        evidence_span_token_recall,
        evidence_span_recall,
    ) = calculate_span_token_recall_and_evidence_span_token_recall_from_dataframe(
        df, decision_boundary
    )
    auprc_score = calculate_auprc(df)
    iou = calculate_iou(df)
    return {
        "decision_boundary": decision_boundary,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auprc": auprc_score,
        "average_number_of_predicted_tokens": average_number_of_predicted_tokens,
        "evidence_span_token_recall": evidence_span_token_recall,
        "evidence_span_recall": evidence_span_recall,
        "precision@1": precision_1,
        "recall@1": recall_1,
        "precision@5": precision_5,
        "recall@5": recall_5,
        "precision@10": precision_10,
        "recall@10": recall_10,
        "precision@50": precision_50,
        "recall@50": recall_50,
        "precision@100": precision_100,
        "recall@100": recall_100,
        "iou": iou,
        "kl_divergence": kl_divergence,
    }


def find_explanation_decision_boundary(df: pl.DataFrame) -> float:
    """Find the decision boundary that maximizes the F1 score.

    Args:
        df (pl.DataFrame): A dataframe with attributions and one hot encoded evidence.

    Returns:
        float: The decision boundary that maximizes the F1 score.
    """
    decision_boundaries = np.linspace(0, 0.1, 101)
    f1_scores = np.zeros_like(decision_boundaries)
    for idx, decision_boundary in enumerate(decision_boundaries):
        df_db = calculate_error_matrix_from_dataframe(df, decision_boundary)
        _, _, f1 = calculate_evidence_prediction_metrics_from_dataframe(df_db)
        f1_scores[idx] = f1
    return decision_boundaries[np.argmax(f1_scores)]


def evaluate_plausibility_and_sparsity(
    model: torch.nn.Module,
    model_path: Path,
    datasets: DatasetDict,
    text_tokenizer: AutoTokenizer,
    target_tokenizer: TargetTokenizer,
    decision_boundary: float,
    cache_explanations: bool,
    explainability_methods: list[str],
    save_path: Path,
) -> None:
    """Calculate the plausibility and sparsity metrics for a given model and explainability methods. Save the results to a csv file.

    Args:
        model (torch.nn.Module): Model to evaluate.
        datasets (DatasetDict): Datasets.
        text_tokenizer (AutoTokenizer): Text tokenizer.
        target_tokenizer (TargetTokenizer): Target tokenizer.
        decision_boundary (float): Decision boundary.
        explainability_methods (list[str]): List of explainability methods to evaluate.
        save_path (Path): Path to save the results.

    """
    schema = {
        "explainability_method": str,
        "prediction_split": str,
        "decision_boundary": float,
        "precision": float,
        "recall": float,
        "f1": float,
        "auprc": float,
        "no_predictions": float,
        "average_number_of_predicted_tokens": float,
        "evidence_span_token_recall": float,
        "evidence_span_recall": float,
        "precision@1": float,
        "recall@1": float,
        "precision@5": float,
        "recall@5": float,
        "precision@10": float,
        "recall@10": float,
        "precision@50": float,
        "recall@50": float,
        "precision@100": float,
        "recall@100": float,
        "entropy": float,
        "kl_divergence": float,
        "iou": float,
        "attributions_sum_zero": float,
    }
    results = pl.DataFrame(schema=schema)

    for explainability_method in explainability_methods:
        pprint("Evaluating " + explainability_method)
        explainer = get_explainability_method(explainability_method)
        explainer_callable = explainer(
            model=model,
            baseline_token_id=text_tokenizer.mask_token_id,
            cls_token_id=text_tokenizer.cls_token_id,
            eos_token_id=text_tokenizer.eos_token_id,
        )

        explanations_val_df = get_explanations(
            model=model,
            model_path=model_path,
            dataset=datasets["validation"],
            explainer=explainer_callable,
            target_tokenizer=target_tokenizer,
            decision_boundary=decision_boundary,
            cache_path=Path(".cache"),
            cache=cache_explanations,
            overwrite_cache=False,
        )
        explanations_val_df = explanations_val_df.filter(
            pl.col("evidence_token_ids").list.len() > 0
        )  # filter out empty groundtruth explanations. They are empty when the evidence is in the truncated text

        # remove start and end tokens from the attributions
        explanations_val_df = explanations_val_df.with_columns(
            pl.col("attributions").map_elements(lambda x: x[1:-1])
        )
        # shift evidence token ids by 1 to account for the removed start token
        explanations_val_df = explanations_val_df.with_columns(
            evidence_token_ids=pl.col("evidence_token_ids").map_elements(
                lambda x: [i - 1 for i in x]
            )
        )

        explanations_val_df = explanations_val_df.with_columns(
            pl.col("attributions").map_elements(
                lambda x: (np.array(x) / (sum(x) + 1e-11)).tolist()
            )
        )

        explanations_test_df = get_explanations(
            model=model,
            model_path=model_path,
            dataset=datasets["test"],
            explainer=explainer_callable,
            target_tokenizer=target_tokenizer,
            cache_path=Path(".cache"),
            cache=cache_explanations,
            overwrite_cache=False,
            decision_boundary=decision_boundary,
        )
        explanations_test_df = explanations_test_df.filter(
            pl.col("evidence_token_ids").list.len() > 0
        )  # filter out empty groundtruth explanations. They are empty when the evidence is in the truncated text

        # remove start and end tokens from the attributions
        explanations_test_df = explanations_test_df.with_columns(
            pl.col("attributions").map_elements(lambda x: x[1:-1])
        )
        # shift evidence token ids by 1 to account for the removed start token
        explanations_test_df = explanations_test_df.with_columns(
            evidence_token_ids=pl.col("evidence_token_ids").map_elements(
                lambda x: [i - 1 for i in x]
            )
        )

        explanations_test_df = explanations_test_df.with_columns(
            pl.col("attributions").map_elements(
                lambda x: (np.array(x) / (sum(x) + 1e-11)).tolist()
            )
        )

        explanation_decision_boundary = find_explanation_decision_boundary(
            explanations_val_df
        )  # use validation set to find the decision boundary

        for prediction_split in ["all", "predicted", "not_predicted"]:
            if prediction_split == "all":
                temp_explanations_test_df = explanations_test_df
            elif prediction_split == "predicted":
                temp_explanations_test_df = explanations_test_df.filter(
                    pl.col("y_prob") > decision_boundary
                )
            elif prediction_split == "not_predicted":
                temp_explanations_test_df = explanations_test_df.filter(
                    pl.col("y_prob") < decision_boundary
                )

            results_dict: dict[str, str | float] = calculate_plausibility_metrics(
                temp_explanations_test_df, explanation_decision_boundary
            )
            results_dict.update(
                calculate_sparsity_metrics(
                    temp_explanations_test_df, explanation_decision_boundary
                )
            )
            results_dict["explainability_method"] = explainability_method
            results_dict["prediction_split"] = prediction_split
            pprint(results_dict)
            results = pl.concat([results, pl.from_dict(results_dict, schema=schema)])

    results.write_csv(save_path)

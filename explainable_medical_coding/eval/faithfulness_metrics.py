# ruff: noqa: E402
import math
from pathlib import Path
from typing import Optional

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

import numpy as np
import polars as pl
import scipy.stats
import torch
from datasets import DatasetDict
from rich.pretty import pprint
from rich.progress import track
from transformers import AutoTokenizer

from explainable_medical_coding.config.factories import get_explainability_method
from explainable_medical_coding.explainability.helper_functions import (
    PermuteDataset,
    create_attention_mask,
)
from explainable_medical_coding.utils.tokenizer import (
    TargetTokenizer,
    get_word_map_roberta,
)
from explainable_medical_coding.utils.analysis import get_explanations, predict
from explainable_medical_coding.utils.loss_functions import (
    one_hot_encode_evidence_token_ids,
)


def sensitivity_n(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    attributions: torch.Tensor,
    device: str | torch.device = "cpu",
    max_n_rate: float = 0.8,
    num_samples_per_n: int = 100,
) -> np.ndarray:
    """Calculate sensitivity-n scores.

    Args:
        model (torch.nn.Module): Model to explain
        input_ids (torch.Tensor): Input token ids [batch_size, sequence_length]
        target_ids (torch.Tensor): Target token ids [num_classes]
        attributions (torch.Tensor): Feature attributions [sequence_length, num_classes]
        max_n_rate (float, optional): Maximum value of n as a fraction of sequence_length. Defaults to 0.8.
        num_samples_per_n (int, optional): Number of samples to draw for each n. Defaults to 100.

    Returns:
        np.ndarray: Pearson correlation scores per class and per n. [max_n, num_classes]
    """
    # assert that batch size is 1
    assert input_ids.shape[0] == 1, "Batch size must be 1."

    model = model.to(device)
    sequence_length = input_ids.shape[1]
    num_classes = target_ids.shape[0]
    max_n = math.ceil(sequence_length * max_n_rate)
    pearson_scores_list = []
    max_n = 10

    # Get indices of sorted attributions (descending).
    input_ids = input_ids.to(device)
    attention_mask = create_attention_mask(input_ids)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        y_pred = (
            model(input_ids, attention_mask)[:, target_ids].detach().cpu()
        )  # [batch_size, num_classes]

    att_sums = np.zeros((max_n - 1, num_samples_per_n, num_classes))
    pred_deltas = np.zeros((max_n - 1, num_samples_per_n, num_classes))

    for _ in range(10):
        for n in track(range(1, max_n), description="Calculating sensitivity-n scores"):
            for sample_idx in range(num_samples_per_n):
                # Sample n indices from the sorted indices.
                n_indices = np.random.choice(
                    np.arange(sequence_length), size=n, replace=False
                )

                # all indices not in n_indices
                set_indices = torch.tensor(
                    np.delete(np.arange(sequence_length), n_indices, axis=0),
                    device=device,
                ).sort()[0]
                input_ids_perturbed = input_ids.clone().index_select(1, set_indices)

                # predict with perturbed input
                attention_mask = create_attention_mask(input_ids_perturbed)
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=True
                ):
                    y_pred_perturb = (
                        model(input_ids_perturbed, attention_mask)[:, target_ids]
                        .detach()
                        .cpu()
                    )  # [batch_size, num_classes]

                # Calculate attribution sum for the perturbed input.
                att_sums[n - 1, sample_idx, :] = attributions.numpy()[n_indices].sum(0)
                pred_deltas[n - 1, sample_idx, :] = (
                    (y_pred - y_pred_perturb).squeeze(0).numpy()
                )

        # Calculate pearson correlation scores per class and per n.
        pearson_scores = np.zeros((max_n - 1, num_classes))
        for n in range(1, max_n):
            for c in range(num_classes):
                pearson_scores[n - 1, c] = scipy.stats.pearsonr(
                    att_sums[n - 1, :, c], pred_deltas[n - 1, :, c]
                )[0]
        pearson_scores_list.append(pearson_scores)
    return pearson_scores


def pertubation_score(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    attributions: torch.Tensor,
    device: str | torch.device = "cpu",
    feature_map: Optional[torch.Tensor] = None,
    baseline_token_id: Optional[int] = None,
    batch_size: int = 32,
    step_size: int = 10,
    num_workers: int = 0,
    descending: bool = True,
    max_features: Optional[int] = 100,
) -> np.ndarray:
    """Calculate pertubation scores for each class. Calculates comprehensiveness scores if descending is True, and sufficiency if False.

    Args:
        model (torch.nn.Module): Model to explain
        input_ids (torch.Tensor): Input token ids [batch_size, sequence_length]
        target_ids (torch.Tensor): Target token ids [num_classes]
        attributions (torch.Tensor): Feature attributions [sequence_length, num_classes]
        device (str | torch.device, optional): Device to run the model on. Defaults to "cpu".
        feature_map (Optional[torch.Tensor], optional): Feature map of shape [sequence_length]. Maps each token to a feature. Defaults to None.
        baseline_token_id (Optional[int], optional): Baseline token id. Defaults to None.
        batch_size (int, optional): Batch size for the dataloader. Defaults to 32.
        step_size (int, optional): Step size for the pertubation. Defaults to 1.
        num_workers (int, optional): Number of workers for the dataloader. Defaults to 2.
        descending (int, optional): Whether to calculate comprehensiveness scores (True) or sufficiency scores (False). Defaults to True.


    Returns:
        torch.Tensor: Pertubation scores of shape [num_features, num_classes]
    """
    # assert that batch size is 1
    assert input_ids.shape[0] == 1, "Batch size must be 1."

    model = model.to(device)
    model.eval()
    model.zero_grad()
    torch.cuda.empty_cache()

    num_classes = target_ids.shape[0]

    dataset = PermuteDataset(
        input_ids=input_ids,
        attributions=attributions,
        feature_map=feature_map,
        baseline_token_id=baseline_token_id,
        step_size=step_size,
        descending=descending,
        max_features=max_features,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.custom_collate_fn,
        num_workers=num_workers,
    )

    attention_mask = create_attention_mask(input_ids)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            logits = (
                model(input_ids, attention_mask).detach().cpu().squeeze(0)
            )  # [num_classes]
            y_prob = torch.sigmoid(logits)

    y_delta = np.zeros((len(dataset), num_classes))
    for target_index, target in enumerate(
        target_ids,
    ):
        dataloader.dataset.set_target_id(target_id=target_index)
        for permuted_input_ids, element_indices in dataloader:
            batch_size = permuted_input_ids.shape[0]
            # number of features not equal 1
            permuted_input_ids = permuted_input_ids.to(device)
            # forward pass with permuted input
            attention_mask = create_attention_mask(permuted_input_ids)
            with torch.no_grad():
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    logits_permute = (
                        model(permuted_input_ids, attention_mask)[:, target]
                        .detach()
                        .cpu()
                        .squeeze(0)
                    )
                    y_prob_permute = torch.sigmoid(logits_permute)
            # Calculate the difference between the prediction with the original input and the prediction with the permuted input
            y_delta[element_indices, target_index] = (
                y_prob[target].repeat(batch_size) - y_prob_permute
            )
    return y_delta


def polarity_score(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    attributions: torch.Tensor,
    device: str | torch.device = "cpu",
    feature_map: Optional[torch.Tensor] = None,
    baseline_token_id: Optional[int] = 500001,
    batch_size: int = 32,
    num_workers: int = 0,
    max_features: Optional[int] = 100,
) -> np.ndarray:
    """Calculate pertubation scores for each class.

    Args:
        model (torch.nn.Module): Model to explain
        input_ids (torch.Tensor): Input token ids [batch_size, sequence_length]
        target_ids (torch.Tensor): Target token ids [num_classes]
        attributions (torch.Tensor): Feature attributions [sequence_length, num_classes]
        device (str | torch.device, optional): Device to run the model on. Defaults to "cpu".
        feature_map (Optional[torch.Tensor], optional): Feature map of shape [sequence_length]. Maps each token to a feature. Defaults to None.
        baseline_token_id (Optional[int], optional): Baseline token id. Defaults to 50001.
        batch_size (int, optional): Batch size for the dataloader. Defaults to 32.
        num_workers (int, optional): Number of workers for the dataloader. Defaults to 2.


    Returns:
        torch.Tensor: Polarity scores of shape [num_features, num_classes]
    """
    # assert that batch size is 1
    assert input_ids.shape[0] == 1, "Batch size must be 1."

    model = model.to(device)
    model.eval()
    model.zero_grad()
    torch.cuda.empty_cache()

    num_classes = target_ids.shape[0]

    dataset = PermuteDataset(
        input_ids=input_ids,
        attributions=attributions,
        feature_map=feature_map,
        baseline_token_id=baseline_token_id,
        step_size=1,
        descending=True,
        cumulative=False,
        max_features=max_features,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.custom_collate_fn,
        num_workers=num_workers,
    )

    attention_mask = create_attention_mask(input_ids)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            logits = (
                model(input_ids, attention_mask).detach().cpu().squeeze(0)
            )  # [num_classes]
            y_prob = torch.sigmoid(logits)

    y_delta = np.zeros((len(dataset), num_classes))
    for target_index, target in enumerate(
        target_ids,
    ):
        dataloader.dataset.set_target_id(target_id=target_index)
        for permuted_input_ids, element_indices in dataloader:
            batch_size = permuted_input_ids.shape[0]
            # number of features not equal 1
            permuted_input_ids = permuted_input_ids.to(device)
            # forward pass with permuted input
            attention_mask = create_attention_mask(permuted_input_ids)
            with torch.no_grad():
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    logits_permute = (
                        model(permuted_input_ids, attention_mask)[:, target]
                        .detach()
                        .cpu()
                        .squeeze(0)
                    )
                    y_prob_permute = torch.sigmoid(logits_permute)
            # Calculate the difference between the prediction with the original input and the prediction with the permuted input
            y_delta[element_indices, target_index] = (
                y_prob[target].repeat(batch_size) - y_prob_permute
            )
    return y_delta


def evaluate_pertubation_scores(
    model: torch.nn.Module,
    dataset: torch.utils.data.DataLoader,
    explanation_df: pl.DataFrame,
    text_tokenizer: AutoTokenizer,
    device: str | torch.device = "cpu",
    batch_size: int = 16,
    word_level: bool = True,
) -> pl.DataFrame:
    """Calculate comprehensiveness and sufficiency scores.

    Args:
        model (torch.nn.Module): Model to explain
        dataloader (torch.utils.data.DataLoader): Dataloader
        explanation_df (pl.DataFrame): Dataframe with explanations
        text_tokenizer (AutoTokenizer): Text tokenizer
        device (str | torch.device, optional): Device. Defaults to "cpu".
        decision_boundary (Optional[float], optional): Decision boundary. Defaults to None.
        step_size (int, optional): Number of words to remove at each step. Defaults to 10.

    Returns:
        tuple[float, float]: Comprehensiveness and sufficiency scores
    """
    schema = {
        "note_id": pl.Utf8,
        "target_id": pl.Int64,
        "y_prob": pl.Float64,
        "comprehensiveness_scores": pl.List(pl.Float64),
        "sufficiency_scores": pl.List(pl.Float64),
        # "polarity_scores": pl.List(pl.Float64),
    }
    rows = []
    for example_idx in track(
        range(len(dataset)), description="Calculating faithfulness"
    ):
        note_id = dataset["note_id"][example_idx]
        input_ids = dataset["input_ids"][example_idx].to(device).unsqueeze(0)
        explanation_df_note = explanation_df.filter((pl.col("note_id") == note_id))
        if len(explanation_df_note) == 0:
            continue

        target_ids = explanation_df_note["target_id"].to_numpy()

        y_prob = torch.tensor(explanation_df_note["y_prob"].to_numpy())
        attributions = torch.tensor(
            explanation_df_note["attributions"].to_list()
        ).T  # [sequence_length, num_classes]

        if word_level:
            word_map = get_word_map_roberta(input_ids, text_tokenizer)
        else:
            word_map = None

        comprehensiveness_scores = pertubation_score(
            model=model,
            input_ids=input_ids,
            target_ids=target_ids,
            attributions=attributions,
            device=device,
            feature_map=word_map,
            baseline_token_id=text_tokenizer.mask_token_id,
            step_size=2,
            batch_size=batch_size,
            num_workers=0,
            descending=True,
            max_features=100,
        )  # [num_features, num_classes]
        sufficiency_scores = pertubation_score(
            model=model,
            input_ids=input_ids,
            target_ids=target_ids,
            attributions=attributions,
            device=device,
            feature_map=word_map,
            baseline_token_id=text_tokenizer.mask_token_id,
            step_size=2,
            batch_size=batch_size,
            num_workers=0,
            descending=False,
            max_features=100,
        )  # [num_features, num_classes]

        # polarity_scores = polarity_score(
        #     model=model,
        #     input_ids=input_ids,
        #     target_ids=target_ids,
        #     attributions=attributions,
        #     device=device,
        #     feature_map=word_map,
        #     baseline_token_id=text_tokenizer.mask_token_id,
        #     batch_size=batch_size,
        #     num_workers=0,
        #     max_features=10,
        # )  # [num_features, num_classes]

        for idx, target_id in enumerate(target_ids):
            row = [
                note_id,
                int(target_id),
                y_prob[idx].item(),
                comprehensiveness_scores[:, idx].tolist(),
                sufficiency_scores[:, idx].tolist(),
                # polarity_scores[:, idx].tolist(),
            ]
            rows.append(row)

        # comprehensiveness_score_sum += (comprehensiveness_scores / y_prob).mean(0).sum()
        # sufficiency_score_sum += (
        #     (np.clip(sufficiency_scores, a_min=0, a_max=100) / y_prob).mean(0).sum()
        # )

    return pl.DataFrame(schema=schema, data=rows)


def evaluate_faithfulness(
    model: torch.nn.Module,
    model_path: Path,
    datasets: DatasetDict,
    text_tokenizer: AutoTokenizer,
    target_tokenizer: TargetTokenizer,
    decision_boundary: float,
    explainability_methods: list[str],
    batch_size: int,
    save_path: Path,
    cache_explanations: bool = True,
    split: str = "test",
) -> None:
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    schema = {
        "explainability_method": str,
        "comprehensiveness": float,
        "sufficiency": float,
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
            (pl.col("y_prob") > decision_boundary)
        )

        faithfulness_df = evaluate_pertubation_scores(
            model,
            datasets[split],
            explanations_test_df,
            text_tokenizer,
            device=device,
            batch_size=batch_size,
            word_level=True,
        )

        sufficiency_scores = faithfulness_df.select(
            pl.struct("sufficiency_scores", "y_prob").map_elements(
                lambda row: np.mean(
                    np.clip(row["sufficiency_scores"], a_min=0, a_max=100)
                )
                / row["y_prob"]
            )
        )
        comprehensiveness_scores = faithfulness_df.select(
            pl.struct("comprehensiveness_scores", "y_prob").map_elements(
                lambda row: np.mean(row["comprehensiveness_scores"]) / row["y_prob"]
            )
        )

        results_dict = {
            "explainability_method": explainability_method,
            "comprehensiveness": float(
                comprehensiveness_scores.mean().to_numpy().squeeze()
            ),
            "sufficiency": float(sufficiency_scores.mean().to_numpy().squeeze()),
        }
        print(results_dict)
        results = pl.concat([results, pl.from_dict(results_dict, schema=schema)])
        faithfulness_df.write_ipc(
            save_path / f"{explainability_method}_faithfulness.arrow"
        )
    results.write_csv(save_path / "faithfulness.csv")


def evaluate_ground_truth_faithfulness(
    model: torch.nn.Module,
    datasets: DatasetDict,
    text_tokenizer: AutoTokenizer,
    batch_size: int,
    save_path: Path,
    split: str = "test",
    word_level: bool = False,
) -> None:
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    dataset = datasets[split]
    dataset = dataset.filter(lambda x: len(x["evidence_input_ids"]) > 0)

    schema = {
        "note_id": pl.Utf8,
        "target_id": pl.Int64,
        "y_prob": pl.Float64,
        "comprehensiveness_scores": pl.List(pl.Float64),
    }
    rows = []
    for example_idx in track(
        range(len(dataset)), description="Calculating faithfulness"
    ):
        note_id = dataset["note_id"][example_idx]
        input_ids = dataset["input_ids"][example_idx].to(device).unsqueeze(0)
        target_ids = dataset["target_ids"][example_idx]
        evidence_input_ids = dataset["evidence_input_ids"][example_idx]
        y_probs = predict(model, input_ids, device).cpu()[0]
        sequence_length = input_ids.shape[1]

        one_hot_evidence = (
            one_hot_encode_evidence_token_ids(evidence_input_ids, sequence_length)
            .to(input_ids.device)
            .T
        )  # [sequence_length, num_classes]

        if word_level:
            word_map = get_word_map_roberta(input_ids, text_tokenizer)
        else:
            word_map = None

        comprehensiveness_scores = pertubation_score(
            model=model,
            input_ids=input_ids,
            target_ids=target_ids,
            attributions=one_hot_evidence,
            device=device,
            feature_map=word_map,
            baseline_token_id=text_tokenizer.mask_token_id,
            step_size=2,
            batch_size=batch_size,
            num_workers=0,
            descending=True,
            max_features=100,
        )  # [num_features, num_classes]

        for idx, target_id in enumerate(target_ids):
            row = [
                note_id,
                int(target_id),
                y_probs[target_id].item(),
                comprehensiveness_scores[:, idx].tolist(),
            ]
            rows.append(row)

    faithfulness_df = pl.DataFrame(schema=schema, data=rows)

    comprehensiveness_scores = faithfulness_df.select(
        pl.struct("comprehensiveness_scores", "y_prob").map_elements(
            lambda row: np.mean(row["comprehensiveness_scores"]) / row["y_prob"]
        )
    )
    print(comprehensiveness_scores.mean())

    faithfulness_df.write_ipc(save_path / "ground_truth_faithfulness.arrow")

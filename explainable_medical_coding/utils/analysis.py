from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import polars as pl
import torch
from datasets.fingerprint import Hasher
from rich.progress import track
from datasets import Dataset

from explainable_medical_coding.utils.tokenizer import TargetTokenizer
from explainable_medical_coding.utils.settings import ID_COLUMN, TARGET_COLUMN


def one_hot(targets: list[list[str]], target2index: dict[str, int]) -> torch.Tensor:
    number_of_classes = len(target2index)
    output_tensor = torch.zeros((len(targets), number_of_classes))
    for idx, case in enumerate(targets):
        for target in case:
            if target in target2index:
                output_tensor[idx, target2index[target]] = 1
    return output_tensor.long()


def load_results(
    run_id: str, experiment_dir: Path, split: str = "val"
) -> tuple[np.array, np.array, np.array, list[str]]:
    print(experiment_dir)
    results = pd.read_feather(experiment_dir / run_id / f"predictions_{split}.feather")
    targets = results[[TARGET_COLUMN]].values.squeeze()
    ids = results[ID_COLUMN].values
    logits_columns = results.drop(columns=[ID_COLUMN, TARGET_COLUMN])
    logits = logits_columns.values
    unique_targets = list(logits_columns.columns.unique())
    return logits, targets, ids, unique_targets


def parse_results(
    logits: np.array, targets: np.array, ids: np.array, unique_targets: list[str]
) -> tuple[dict[str, torch.Tensor], str, int, dict[str, int]]:
    target2index = {target: idx for idx, target in enumerate(unique_targets)}
    # Mapping from target to index and vice versa
    targets = one_hot(targets, target2index)  # one_hot encoding of targets
    logits = torch.tensor(logits)
    return logits, targets, ids, target2index


def get_results(
    run_id: str, experiment_dir: Path, split: str = "val"
) -> tuple[torch.Tensor, torch.Tensor, np.array, dict[str, int]]:
    logits, targets, ids, unique_targets = load_results(run_id, experiment_dir, split)
    logits, targets, ids, target2index = parse_results(
        logits, targets, ids, unique_targets
    )
    return logits, targets, ids, target2index


def get_target_counts(targets: list[list[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in targets:
        for target in case:
            if target in counts:
                counts[target] += 1
            else:
                counts[target] = 1
    return counts


def f1_score_db_tuning(logits, targets, average="micro", type="single"):
    if average not in ["micro", "macro"]:
        raise ValueError("Average must be either 'micro' or 'macro'")
    dbs = torch.linspace(0, 1, 100)
    tp = torch.zeros((len(dbs), targets.shape[1]))
    fp = torch.zeros((len(dbs), targets.shape[1]))
    fn = torch.zeros((len(dbs), targets.shape[1]))
    for idx, db in enumerate(dbs):
        predictions = (logits > db).long()
        tp[idx] = torch.sum((predictions) * (targets), dim=0)
        fp[idx] = torch.sum(predictions * (1 - targets), dim=0)
        fn[idx] = torch.sum((1 - predictions) * targets, dim=0)
    if average == "micro":
        f1_scores = tp.sum(1) / (tp.sum(1) + 0.5 * (fp.sum(1) + fn.sum(1)) + 1e-10)
    else:
        f1_scores = torch.mean(tp / (tp + 0.5 * (fp + fn) + 1e-10), dim=1)
    if type == "single":
        best_f1 = f1_scores.max()
        best_db = dbs[f1_scores.argmax()]
        print(f"Best F1: {best_f1:.4f} at DB: {best_db:.4f}")
        return best_f1, best_db
    if type == "per_class":
        best_f1 = f1_scores.max(1)
        best_db = dbs[f1_scores.argmax(0)]
        print(f"Best F1: {best_f1} at DB: {best_db}")
        return best_f1, best_db


def micro_f1(pred: torch.Tensor, targets: torch.Tensor) -> float:
    tp = torch.sum((pred) * (targets), dim=0)
    fp = torch.sum(pred * (1 - targets), dim=0)
    fn = torch.sum((1 - pred) * targets, dim=0)
    f1 = tp / (tp + 0.5 * (fp + fn) + 1e-10)
    return torch.mean(f1)


def get_db(run_id: str, experiment_dir: Path) -> torch.Tensor:
    logits, targets, _, _ = get_results(run_id, experiment_dir, "val")
    _, db = f1_score_db_tuning(logits, targets, average="micro")
    return db


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1.0 - x))


@torch.no_grad()
def get_probs_and_targets(
    model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, cache: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cache_path = Path("cache") / f"{Hasher.hash(model)}_{Hasher.hash(dataloader)}.pt"

    if cache_path.exists() and cache:
        return torch.load(cache_path)

    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    y_probs_list = []
    targets_list = []
    loss_list = []
    ids_list = []

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        for batch in track(dataloader):
            batch = batch.to(device)
            input_ids, attention_mask, targets = (
                batch.input_ids,
                batch.attention_masks,
                batch.targets,
            )
            logits = model(input_ids, attention_mask)
            y_probs = torch.sigmoid(logits)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
            y_probs_list.append(y_probs.cpu())
            targets_list.append(targets.cpu())
            loss_list.append(loss.item())
            ids_list += batch.ids

    y_probs = torch.cat(y_probs_list)  # [examples, classes]
    targets = torch.cat(targets_list)  # [examples, classes]
    loss = torch.tensor(loss_list)  # [examples]

    cache_path.parent.mkdir(exist_ok=True)
    torch.save((y_probs, targets, loss, ids_list), cache_path)

    return y_probs, targets, loss, ids_list


def get_explanations(
    model: torch.nn.Module,
    model_path: Path,
    dataset: Dataset,
    explainer: Callable,
    target_tokenizer: TargetTokenizer,
    cache_path: Path = Path(".cache"),
    pad_token_id: int = 1,
    cache: bool = True,
    overwrite_cache: bool = False,
    decision_boundary: float = 0.5,
) -> pl.DataFrame:
    """This function calculates the explanations for a given model and dataloader and save it in a dataframe. The dataframe is cached in the cache folder.

    Args:
        model (torch.nn.Module): Model to explain
        dataset (Dataset): Dataset to explain
        explainer (Callable): Explainer callable
        target_tokenizer (TargetTokenizer): Target tokenizer
        cache_path (Path, optional): Path to cache folder. Defaults to Path(".cache").
        pad_token_id (int, optional): Padding token id. Defaults to 1.
        decision_boundary (float, optional): Decision boundary. Defaults to 0.5.

    Returns:
        pl.DataFrame: Dataframe with explanations.
    """
    cache_path = (
        cache_path
        / f"{Hasher.hash(model_path)}_{Hasher.hash(explainer)}_{Hasher.hash(dataset)}_{Hasher.hash(decision_boundary)}.parquet"
    )

    if cache_path.exists() and cache:
        return pl.read_parquet(cache_path)

    schema = {
        "note_id": pl.Utf8,
        "target_id": pl.Int64,
        "y_prob": pl.Float64,
        "attributions": pl.List(pl.Float64),
        "evidence_token_ids": pl.List(pl.Int64),
    }

    rows = []
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"

    for example_idx in track(range(len(dataset))):
        note_id = dataset["note_id"][example_idx]
        input_ids = dataset["input_ids"][example_idx].to(device).unsqueeze(0)
        ground_truth_target_ids = dataset["target_ids"][example_idx].tolist()
        evidence_input_ids = dataset["evidence_input_ids"][example_idx]
        target_id2evidence_input_ids = {
            ground_truth_target_id: evidence_input_ids[idx]
            for idx, ground_truth_target_id in enumerate(ground_truth_target_ids)
        }

        y_probs = predict(model, input_ids, device).cpu()[0]
        predicted_target_ids = torch.where(y_probs > decision_boundary)[0].tolist()
        target_ids = torch.tensor(
            list(set(ground_truth_target_ids) | set(predicted_target_ids))
        )

        attributions = explainer(
            input_ids=input_ids,
            target_ids=target_ids,
            device=device,
        )  # [sequence_length, num_classes]

        for idx, target_id in enumerate(target_ids):
            row = [
                note_id,
                target_id.item(),
                y_probs[target_id].item(),
                attributions[:, idx].tolist(),
                target_id2evidence_input_ids.get(target_id.item(), []),
            ]
            rows.append(row)

    cache_path.parent.mkdir(exist_ok=True)
    df = pl.DataFrame(schema=schema, data=rows)
    if cache or (overwrite_cache and not cache):
        df.write_parquet(cache_path)
    return df


def create_attention_mask(input_ids: torch.Tensor) -> torch.Tensor:
    """Create attention mask for a given input

    Args:
        input_ids (torch.Tensor): Input ids to create attention mask for

    Returns:
        torch.Tensor: Attention mask
    """
    attention_mask = torch.ones_like(input_ids)
    return attention_mask


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    input_ids: torch.Tensor | list[torch.Tensor],
    device: str | torch.device,
    return_logits: bool = False,
    pad_id: int = 1,
) -> torch.Tensor:
    """Output a model's predictions

    Args:
        model (torch.nn.Module): Model to predict with
        input_ids (torch.Tensor): Input ids to predict on
        device (str | torch.device): Device to run the model on
        return_logits (bool, optional): Whether to return logits or not (sigmoid probabilities). Defaults to False.

    Returns:
        torch.Tensor: Predictions
    """
    if isinstance(input_ids, list):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_id
        )
    elif isinstance(input_ids, torch.Tensor):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
        elif len(input_ids.shape) > 2:
            raise ValueError("Input ids must be 1D or 2D")
    else:
        raise ValueError("Input ids must be a tensor or a list of tensors")

    input_ids = input_ids.to(device)
    attention_mask = create_attention_mask(input_ids)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        logits = model(input_ids, attention_mask).detach()
    if return_logits:
        return logits
    return torch.sigmoid(logits)

from typing import Any, Optional

import torch
from transformers import PreTrainedTokenizer

from datasets import DatasetDict
from explainable_medical_coding.utils.settings import TARGET_COLUMN
from explainable_medical_coding.utils.datatypes import Lookups
from explainable_medical_coding.utils.tokenizer import TargetTokenizer


def create_lookups(
    dataset: DatasetDict,
    text_tokenizer: PreTrainedTokenizer,
    target_tokenizer: TargetTokenizer,
) -> Lookups:
    """Load the lookups.

    Args:
        dataframe (pd.DataFrame): The dataframe.
        tokenizer (PreTrainedTokenizer): The tokenizer.

    Returns:
        Lookups: The lookups.
    """

    split2code_indices = create_split2target_indices_lookup(
        dataset, target_tokenizer=target_tokenizer
    )
    data_info = get_data_info(
        dataset=dataset,
        vocab_size=len(text_tokenizer),
        pad_token_id=text_tokenizer.pad_token_id,
        pad_target_id=target_tokenizer.pad_id,
        sos_target_id=target_tokenizer.sos_id,
        eos_target_id=target_tokenizer.eos_id,
        num_classes=len(target_tokenizer),
        split2code_indices=split2code_indices,
    )

    return Lookups(
        data_info=data_info,
        split2code_indices=split2code_indices,
        target_tokenizer=target_tokenizer,
    )


def create_split2target_indices_lookup(
    dataset: DatasetDict, target_tokenizer: TargetTokenizer
) -> dict[str, torch.Tensor]:
    split2code_indices = {}
    for split_name, data in dataset.items():
        unique_codes = (
            data.with_format("pandas")[TARGET_COLUMN].explode().unique().tolist()
        )
        target_ids = target_tokenizer(unique_codes)
        split2code_indices[split_name] = torch.tensor(target_ids)
    split2code_indices["train_val"] = split2code_indices["train"]
    return split2code_indices


def get_data_info(
    dataset: DatasetDict,
    vocab_size: int,
    pad_token_id: int,
    pad_target_id: Optional[int],
    sos_target_id: Optional[int],
    eos_target_id: Optional[int],
    num_classes: int,
    split2code_indices: dict[str, list],
) -> dict:
    data_info: dict[str, Any] = {}
    data_info["num_examples"] = (
        len(dataset["train"]) + len(dataset["validation"]) + len(dataset["test"])
    )
    data_info["num_train_examples"] = len(dataset["train"])
    data_info["num_val_examples"] = len(dataset["validation"])
    data_info["num_test_examples"] = len(dataset["test"])

    data_info["average_words_per_example"] = (
        dataset["train"].with_format("pandas")["input_ids"].apply(len).mean()
        + dataset["validation"].with_format("pandas")["input_ids"].apply(len).mean()
        + dataset["test"].with_format("pandas")["input_ids"].apply(len).mean()
    ) / 3

    data_info["average_targets_per_example"] = (
        dataset["train"].with_format("pandas")[TARGET_COLUMN].apply(len).mean()
        + dataset["validation"].with_format("pandas")[TARGET_COLUMN].apply(len).mean()
        + dataset["test"].with_format("pandas")[TARGET_COLUMN].apply(len).mean()
    ) / 3

    data_info["num_classes"] = num_classes
    data_info["num_train_classes"] = len(split2code_indices["train"])
    data_info["num_val_classes"] = len(split2code_indices["validation"])
    data_info["num_test_classes"] = len(split2code_indices["test"])
    data_info["vocab_size"] = vocab_size
    data_info["pad_token_id"] = pad_token_id
    data_info["pad_target_id"] = pad_target_id
    data_info["sos_target_id"] = sos_target_id
    data_info["eos_target_id"] = eos_target_id

    return data_info

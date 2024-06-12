# ruff: noqa: E402
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
import torch
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer

import explainable_medical_coding.config.factories as factories
from explainable_medical_coding.utils.tokenizer import TargetTokenizer
from explainable_medical_coding.utils.data_helper_functions import (
    create_targets_column,
    filter_unknown_targets,
    format_evidence_spans,
)
from explainable_medical_coding.utils.settings import TARGET_COLUMN, TEXT_COLUMN


# Load model
def load_trained_model(
    experiment_path: Path,
    config: OmegaConf,
    pad_token_id: int = 1,
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, float]:
    checkpoint = torch.load(experiment_path / "best_model.pt", map_location=device)
    # hacky solution. Should save the number of classes in a better way
    num_classes = checkpoint["num_classes"]
    print(config.model)
    model = factories.get_model(
        config.model, {"num_classes": num_classes, "pad_token_id": pad_token_id}
    )
    model.to(device)

    # torch compile sometimes stores the model in a strange way. this is a hack to fix that
    if "_orig_mod" in list(checkpoint["model"].keys())[0]:
        opt_model = torch.compile(model)
        if "_orig_mod.label_wise_attention.layernorm_1.weight" in checkpoint["model"]:
            del checkpoint["model"]["_orig_mod.label_wise_attention.layernorm_1.weight"]
            del checkpoint["model"]["_orig_mod.label_wise_attention.layernorm_1.bias"]
        if "_orig_mod.roberta_encoder.pooler.dense.weight" in checkpoint["model"]:
            del checkpoint["model"]["_orig_mod.roberta_encoder.pooler.dense.weight"]
            del checkpoint["model"]["_orig_mod.roberta_encoder.pooler.dense.bias"]
        if "_orig_mod.roberta_encoder.embeddings.position_ids" in checkpoint["model"]:
            del checkpoint["model"]["_orig_mod.roberta_encoder.embeddings.position_ids"]

        opt_model.load_state_dict(checkpoint["model"])

    else:
        model.load_state_dict(checkpoint["model"])

    model.eval()

    return model, float(checkpoint["db"].to("cpu"))


def load_and_prepare_dataset(
    dataset_path: Path,
    text_tokenizer: AutoTokenizer,
    target_tokenizer: TargetTokenizer,
    max_input_length: int = 6000,
    target_columns: list[str] = ["diagnosis_codes", "procedure_codes"],
) -> Dataset:
    dataset = load_dataset(str(dataset_path))

    # tokenize text
    dataset = dataset.map(
        lambda x: text_tokenizer(
            x[TEXT_COLUMN],
            return_length=True,
            truncation=True,
            max_length=max_input_length,
        ),
        batched=True,
        num_proc=8,
        batch_size=1_000,
        desc="Tokenizing text",
    )
    dataset = dataset.map(
        lambda x: create_targets_column(x, target_columns),
        desc="Creating targets column",
    )
    # remove targets that are not in the target tokenizer
    print(
        f"Number of test targets before filtering: {len(dataset['test'].with_format('pandas')[TARGET_COLUMN].explode())} "
    )
    dataset = dataset.map(
        lambda x: filter_unknown_targets(x, set(target_tokenizer.target2id.keys())),
        desc="Filter unknown targets",
    )
    print(
        f"Number of test targets after filtering: {len(dataset['test'].with_format('pandas')[TARGET_COLUMN].explode())} "
    )

    # remove cases with no targets
    dataset = dataset.filter(
        lambda x: len(x[TARGET_COLUMN]) > 0, desc="Filtering empty targets"
    )

    # convert targets to IDs
    dataset = dataset.map(
        lambda x: {"target_ids": target_tokenizer(x[TARGET_COLUMN])},
        desc="Converting targets to IDs",
    )

    dataset = dataset.map(lambda x: format_evidence_spans(x, text_tokenizer))

    dataset.set_format(
        type="torch", columns=["input_ids", "length", "attention_mask", "target_ids"]
    )

    return dataset

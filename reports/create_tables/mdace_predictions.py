# ruff: noqa: E402
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
load_dotenv(find_dotenv())

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from datasets import concatenate_datasets
from omegaconf import OmegaConf
from rich.pretty import pprint
from transformers import AutoTokenizer

from explainable_medical_coding.config.factories import (
    get_dataloaders,
    get_metric_collections,
)
from explainable_medical_coding.utils.analysis import get_probs_and_targets
from explainable_medical_coding.utils.data_helper_functions import (
    get_code2description_mimiciii,
)
from explainable_medical_coding.utils.loaders import (
    load_and_prepare_dataset,
    load_trained_model,
)
from explainable_medical_coding.utils.lookups import create_lookups
from explainable_medical_coding.utils.tokenizer import TargetTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COMBINE_TRAIN_TEST = False
EXPERIMENT_PATH = Path("models")

model_run_ids = {
    "B$_{\\text{U}}$": EXPERIMENT_PATH / "unsupervised",
    "B$_{\\text{S}}$": EXPERIMENT_PATH / "supervised",
    "IGR": EXPERIMENT_PATH / "igr",
    "TM": EXPERIMENT_PATH / "tm",
    "PGD": EXPERIMENT_PATH / "pgd",
}
result_dict: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
data_config = OmegaConf.load(
    "explainable_medical_coding/config/data/mdace_inpatient_icd9.yaml"
)

target_columns = list(data_config.target_columns)
dataset_path = Path(data_config.dataset_path)

for model_name, folder_name in model_run_ids.items():
    for experiment_path in folder_name.iterdir():
        config = OmegaConf.load(experiment_path / "config.yaml")
        text_tokenizer_path = config.model.configs.model_path

        # target_tokenizer.load(experiment_path / "target_tokenizer.json")
        target_tokenizer = TargetTokenizer(autoregressive=False)
        target_tokenizer.load(experiment_path / "target_tokenizer.json")

        text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_path)
        max_input_length = int(data_config.max_length)

        dataset = load_and_prepare_dataset(
            dataset_path,
            text_tokenizer,
            target_tokenizer,
            max_input_length,
            target_columns,
        )
        dataset = dataset.filter(
            lambda x: x["note_type"] == "Discharge summary",
            desc="Filtering all notes that are not discharge summaries",
        )
        if COMBINE_TRAIN_TEST:
            dataset["test"] = concatenate_datasets([dataset["train"], dataset["test"]])

        lookups = create_lookups(
            dataset=dataset,
            text_tokenizer=text_tokenizer,
            target_tokenizer=target_tokenizer,
        )

        model, decision_boundary = load_trained_model(
            experiment_path,
            config,
            pad_token_id=text_tokenizer.pad_token_id,
            device=device,
        )

        # code2description = get_code2description_mimiciv()
        code2description = get_code2description_mimiciii()
        code2count = (
            dataset["train"]
            .with_format("pandas")["target"]
            .explode()
            .value_counts()
            .to_dict()
        )

        # use large batch size to speed up evaluation
        config.dataloader.max_batch_size = 64
        config.dataloader.batch_size = 64

        dataloaders = get_dataloaders(
            config=config.dataloader,
            dataset=dataset,
            target_tokenizer=lookups.target_tokenizer,
            pad_token_id=lookups.data_info["pad_token_id"],
        )

        metric_collections = get_metric_collections(
            config=config.metrics,
            number_of_classes=lookups.data_info["num_classes"],
            split2code_indices=lookups.split2code_indices,
            autoregressive=config.model.autoregressive,
        )
        for split_name, metric_collection in metric_collections.items():
            metric_collection.set_threshold(decision_boundary)

        y_probs_test, targets_test, loss_test, ids_test = get_probs_and_targets(
            model, dataloaders["test"], cache=False
        )
        metric_collections["test"].update(
            y_probs=y_probs_test,
            targets=targets_test,
        )
        results = metric_collections["test"].compute()
        for metric_name, metric_value in results.items():
            result_dict[model_name][metric_name].append(metric_value.item())
        pprint(results)

result_dict_str: dict[str, dict[str, str]] = defaultdict(lambda: defaultdict(str))
for model_name in result_dict.keys():
    for metric_name in result_dict[model_name].keys():
        result_dict_str[model_name][
            metric_name
        ] = f"{np.mean(result_dict[model_name][metric_name])*100:.1f} \\pm {np.std(result_dict[model_name][metric_name])*100:.1f}"

print(pd.DataFrame(result_dict_str).T[["f1_micro", "f1_macro", "map"]])

from typing import Optional, Type, Callable
import functools

import torch
import transformers
from omegaconf import OmegaConf
from rich.pretty import pprint

import explainable_medical_coding.data.batch_sampler as batch_samplers
import explainable_medical_coding.eval.metrics as metrics
import explainable_medical_coding.trainer.callbacks as callbacks
from datasets import DatasetDict
from explainable_medical_coding.data.dataloader import BaseDataset
from explainable_medical_coding.models import models
from explainable_medical_coding.trainer import trainer
from explainable_medical_coding.utils.datatypes import Lookups
from explainable_medical_coding.utils.lookups import create_lookups
from explainable_medical_coding.utils.tokenizer import TargetTokenizer
import explainable_medical_coding.explainability.explanation_methods as explainability_methods
import explainable_medical_coding.utils.loss_functions as loss_functions


def get_lookups(
    dataset: DatasetDict,
    text_tokenizer: transformers.PreTrainedTokenizer,
    target_tokenizer: TargetTokenizer,
) -> Lookups:
    return create_lookups(
        dataset=dataset,
        text_tokenizer=text_tokenizer,
        target_tokenizer=target_tokenizer,
    )


def get_model(config: OmegaConf, data_info: dict) -> torch.nn.Module:
    model_class = getattr(models, config.name)
    return model_class(**data_info, **config.configs)


def get_optimizer(config: OmegaConf, model: torch.nn.Module) -> torch.optim.Optimizer:
    optimizer_class = getattr(torch.optim, config.name)
    if config.configs.weight_decay:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": config.configs.weight_decay,
                "lr": config.configs.lr,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": config.configs.lr,
            },
        ]
    else:
        optimizer_grouped_parameters = model.parameters()

    return optimizer_class(optimizer_grouped_parameters, **config.configs)


def get_lr_scheduler(
    config: OmegaConf,
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    config_dict = OmegaConf.to_container(config.configs)  # convert to dict
    if config.name is None:
        return None

    if config.configs.warmup:
        config_dict["num_warmup_steps"] = config.configs.warmup * num_training_steps
        del config_dict["warmup"]

    if hasattr(torch.optim.lr_scheduler, config.name):
        lr_scheduler_class = getattr(torch.optim.lr_scheduler, config.name)
        return lr_scheduler_class(optimizer, **config_dict)
    from transformers import get_scheduler

    return get_scheduler(
        name=config.name,
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        **config_dict,
    )


def get_dataloaders(
    config: OmegaConf,
    dataset: DatasetDict,
    target_tokenizer: TargetTokenizer,
    pad_token_id: int = 1,
) -> dict[str, torch.utils.data.DataLoader]:
    dataloaders = {}
    datasets_dict = {}
    for split_name, data in dataset.items():
        datasets_dict[split_name] = BaseDataset(
            data,
            target_tokenizer=target_tokenizer,
            pad_token_id=pad_token_id,
        )
    datasets_dict["train_val"] = BaseDataset(
        dataset["train"].train_test_split(test_size=0.1)["test"],
        target_tokenizer=target_tokenizer,
        pad_token_id=pad_token_id,
    )
    train_batch_size = min(config.batch_size, config.max_batch_size)
    pprint(f"Train batch size: {train_batch_size}")
    if config.batch_sampler.name:
        batch_sampler_class = getattr(batch_samplers, config.batch_sampler.name)
        batch_sampler = batch_sampler_class(
            dataset=dataset["train"],
            batch_size=train_batch_size,
            drop_last=config.drop_last,
            **config.batch_sampler.configs,
        )
        dataloaders["train"] = torch.utils.data.DataLoader(
            datasets_dict["train"],
            batch_sampler=batch_sampler,
            collate_fn=datasets_dict["train"].collate_fn,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
    else:
        dataloaders["train"] = torch.utils.data.DataLoader(
            datasets_dict["train"],
            shuffle=True,
            batch_size=train_batch_size,
            drop_last=config.drop_last,
            collate_fn=datasets_dict["train"].collate_fn,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
    dataloaders["train_val"] = torch.utils.data.DataLoader(
        datasets_dict["train_val"],
        batch_size=config.max_batch_size,
        shuffle=False,
        collate_fn=datasets_dict["train_val"].collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    dataloaders["validation"] = torch.utils.data.DataLoader(
        datasets_dict["validation"],
        batch_size=config.max_batch_size,
        shuffle=False,
        collate_fn=datasets_dict["validation"].collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    dataloaders["test"] = torch.utils.data.DataLoader(
        datasets_dict["test"],
        batch_size=config.max_batch_size,
        shuffle=False,
        collate_fn=datasets_dict["test"].collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return dataloaders


def get_metric_collection(
    config: OmegaConf,
    number_of_classes: int,
    split_code_indices: Optional[torch.Tensor] = None,
    autoregressive: bool = False,
) -> metrics.MetricCollection:
    metric_list = []
    for metric in config:
        metric_class = getattr(metrics, metric["name"])
        metric_list.append(
            metric_class(number_of_classes=number_of_classes, **metric["configs"])
        )

    if split_code_indices is not None:
        code_indices = split_code_indices.clone()
    else:
        code_indices = None

    return metrics.MetricCollection(
        metrics=metric_list,
        code_indices=code_indices,
        autoregressive=autoregressive,
    )


def get_metric_collections(
    config: OmegaConf,
    number_of_classes: int,
    split_names: list[str] = ["train", "train_val", "validation", "test"],
    split2code_indices: Optional[dict[str, torch.Tensor]] = None,
    autoregressive: bool = False,
) -> dict[str, metrics.MetricCollection]:
    metric_collections: dict[str, metrics.MetricCollection] = {}
    for split_name in split_names:
        if split2code_indices is not None:
            split_code_indices = split2code_indices.get(split_name)
        else:
            split_code_indices = None

        metric_collections[split_name] = get_metric_collection(
            config=config,
            number_of_classes=number_of_classes,
            split_code_indices=split_code_indices,
            autoregressive=autoregressive,
        )
    return metric_collections


def get_callbacks(config: OmegaConf) -> list[callbacks.BaseCallback]:
    callbacks_list = []
    for callback in config:
        callback_class = getattr(callbacks, callback.name)
        callbacks_list.append(callback_class(config=callback.configs))
    return callbacks_list


def get_trainer(name: str) -> Type[trainer.Trainer]:
    return getattr(trainer, name)


def get_explainability_method(name: str) -> Callable:
    methods = {
        "laat": explainability_methods.get_laat_callable,
        "occlusion": explainability_methods.get_occlusion_1_callable,
        "deeplift": explainability_methods.get_deeplift_callable,
        "integrated_gradient": explainability_methods.get_integrated_gradient_callable,
        "gradient_x_input": explainability_methods.get_gradient_x_input_callable,
        "kernelshap": explainability_methods.get_kernelshap_callable,
        "lime": explainability_methods.get_lime_callable,
        "attention_rollout": explainability_methods.get_attention_rollout_callable,
        "alti": explainability_methods.get_alti_callable,
        "random": explainability_methods.get_random_baseline_callable,
        "grad_attention": explainability_methods.get_grad_attention_callable,
        "atgrad_attention": explainability_methods.get_atgrad_attention_callable,
    }

    if name not in methods:
        raise ValueError(
            f"Explainability method {name} not implemented. Select on of the following: {methods.keys()}"
        )

    return methods[name]


def get_loss_function(config: OmegaConf) -> torch.nn.Module:
    return functools.partial(getattr(loss_functions, config.name), **config.configs)

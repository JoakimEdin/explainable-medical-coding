# ruff: noqa: E402
import logging
import math
from pathlib import Path

# load environment variables
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

import hydra
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer

import explainable_medical_coding.config.factories as factories
from explainable_medical_coding.utils.loaders import (
    load_trained_model,
)
from explainable_medical_coding.utils.tokenizer import TargetTokenizer
from explainable_medical_coding.utils.analysis import predict
from explainable_medical_coding.utils.data_helper_functions import (
    create_targets_column,
    filter_unknown_targets,
    format_evidence_spans,
    get_unique_targets,
)
from explainable_medical_coding.utils.seed import set_seed
from explainable_medical_coding.utils.settings import TARGET_COLUMN, TEXT_COLUMN
from explainable_medical_coding.utils.tensor import deterministic, set_gpu

LOGGER = logging.getLogger(name=__file__)
LOGGER.setLevel(logging.INFO)


@hydra.main(
    version_base=None,
    config_path="explainable_medical_coding/config",
    config_name="config",
)
def main(cfg: OmegaConf) -> None:
    if cfg.deterministic:
        deterministic()

    set_seed(cfg.seed)
    device = set_gpu(cfg)

    target_columns = list(cfg.data.target_columns)
    dataset_path = Path(cfg.data.dataset_path)
    model_path = Path(cfg.load_model) if cfg.load_model is not None else None
    dataset = load_dataset(str(dataset_path))

    text_tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.configs.model_path,
    )
    max_input_length = int(cfg.data.max_length)
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
    known_targets = set(get_unique_targets(dataset))
    dataset = dataset.map(
        lambda x: filter_unknown_targets(x, known_targets=known_targets),
        desc="Filter unknown targets",
    )
    dataset = dataset.filter(
        lambda x: len(x[TARGET_COLUMN]) > 0, desc="Filtering empty targets"
    )
    dataset = dataset.map(lambda x: format_evidence_spans(x, text_tokenizer))

    autoregressive = bool(cfg.model.autoregressive)
    target_tokenizer = TargetTokenizer(autoregressive=autoregressive)
    if model_path is None:
        unique_targets = get_unique_targets(dataset)
        target_tokenizer.fit(unique_targets)
    else:
        LOGGER.info("Loading Tokenizer from model_path")
        target_tokenizer.load(model_path / "target_tokenizer.json")

    # convert targets to target ids
    dataset = dataset.map(
        lambda x: {"target_ids": target_tokenizer(x[TARGET_COLUMN])},
        desc="Converting targets to target ids",
    )
    dataset.set_format(
        type="torch", columns=["input_ids", "length", "attention_mask", "target_ids"]
    )

    lookups = factories.get_lookups(
        dataset=dataset,
        text_tokenizer=text_tokenizer,
        target_tokenizer=target_tokenizer,
    )
    LOGGER.info(lookups.data_info)

    if model_path is None:
        model = factories.get_model(config=cfg.model, data_info=lookups.data_info)
    else:
        LOGGER.info("Loading Model from model_path")
        saved_config = OmegaConf.load(model_path / "config.yaml")
        model, decision_boundary = load_trained_model(
            model_path,
            saved_config,
            pad_token_id=text_tokenizer.pad_token_id,
            device=device,
        )

    model.to(device)
    # model = torch.compile(model)

    if cfg.distillation:
        if model_path is None:
            raise ValueError("Distillation requires a pre-trained model")
        dataset = dataset.sort("length")
        model.eval()
        dataset = dataset.map(
            lambda x: {
                "teacher_logits": predict(
                    model,
                    x["input_ids"],
                    device=device,
                    return_logits=True,
                    pad_id=text_tokenizer.pad_token_id,
                ),
            },
            desc="Adding teacher logits",
            batched=True,
            batch_size=64,
        )
        model.train()

    loss_function = factories.get_loss_function(config=cfg.loss)

    dataloaders = factories.get_dataloaders(
        config=cfg.dataloader,
        dataset=dataset,
        target_tokenizer=lookups.target_tokenizer,
        pad_token_id=lookups.data_info["pad_token_id"],
    )

    metric_collections = factories.get_metric_collections(
        config=cfg.metrics,
        number_of_classes=lookups.data_info["num_classes"],
        split2code_indices=lookups.split2code_indices,
        autoregressive=cfg.model.autoregressive,
    )

    optimizer = factories.get_optimizer(config=cfg.optimizer, model=model)
    accumulate_grad_batches = int(
        max(cfg.dataloader.batch_size / cfg.dataloader.max_batch_size, 1)
    )
    num_training_steps = (
        math.ceil(len(dataloaders["train"]) / accumulate_grad_batches)
        * cfg.trainer.epochs
    )
    lr_scheduler = factories.get_lr_scheduler(
        config=cfg.lr_scheduler,
        optimizer=optimizer,
        num_training_steps=num_training_steps,
    )
    callbacks = factories.get_callbacks(config=cfg.callbacks)
    trainer_class = factories.get_trainer(name=cfg.trainer.name)
    trainer = trainer_class(
        config=cfg,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        dataloaders=dataloaders,
        metric_collections=metric_collections,
        callbacks=callbacks,
        lr_scheduler=lr_scheduler,
        lookups=lookups,
        accumulate_grad_batches=accumulate_grad_batches,
    ).to(device)

    trainer.fit()


if __name__ == "__main__":
    main()

# ruff: noqa: E402
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


load_dotenv(find_dotenv())

import hydra
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from datasets import concatenate_datasets

from explainable_medical_coding.eval.faithfulness_metrics import evaluate_faithfulness
from explainable_medical_coding.eval.plausibility_metrics import (
    evaluate_plausibility_and_sparsity,
)
from explainable_medical_coding.utils.loaders import (
    load_and_prepare_dataset,
    load_trained_model,
)
from explainable_medical_coding.utils.tokenizer import TargetTokenizer
from explainable_medical_coding.utils.tensor import set_gpu


@hydra.main(
    version_base=None,
    config_path="explainable_medical_coding/config",
    config_name="explainability",
)
def main(cfg: OmegaConf) -> None:
    device = set_gpu(cfg)
    target_columns = list(cfg.data.target_columns)
    dataset_path = Path(cfg.data.dataset_path)
    model_folder_path = Path(cfg.model_folder_path)
    run_id = cfg.run_id
    model_path = model_folder_path / run_id
    saved_config = OmegaConf.load(model_path / "config.yaml")
    text_tokenizer_path = saved_config.model.configs.model_path

    # target_tokenizer.load(experiment_path / "target_tokenizer.json")
    target_tokenizer = TargetTokenizer(autoregressive=False)
    target_tokenizer.load(model_path / "target_tokenizer.json")

    text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_path)
    max_input_length = int(saved_config.data.max_length)

    dataset = load_and_prepare_dataset(
        dataset_path, text_tokenizer, target_tokenizer, max_input_length, target_columns
    )
    dataset = dataset.filter(
        lambda x: x["note_type"] == "Discharge summary",
        desc="Filtering all notes that are not discharge summaries",
    )

    if cfg.combine_test_train:
        dataset["test"] = concatenate_datasets([dataset["test"], dataset["train"]])

    model, decision_boundary = load_trained_model(
        model_path,
        saved_config,
        pad_token_id=text_tokenizer.pad_token_id,
        device=device,
    )

    results_dir = Path("reports/explainability_results/") / cfg.model_name
    if cfg.create_run_id_folder:
        results_dir = results_dir / run_id

    results_dir.mkdir(parents=True, exist_ok=True)

    evaluate_plausibility_and_sparsity(
        model=model,
        model_path=model_path,
        datasets=dataset,
        text_tokenizer=text_tokenizer,
        target_tokenizer=target_tokenizer,
        decision_boundary=decision_boundary,
        explainability_methods=cfg.explainers,
        cache_explanations=cfg.cache_explanations,
        save_path=results_dir / "plausibility_and_sparsity.csv",
    )
    if cfg.evaluate_faithfulness:
        evaluate_faithfulness(
            model=model,
            model_path=model_path,
            datasets=dataset,
            text_tokenizer=text_tokenizer,
            target_tokenizer=target_tokenizer,
            decision_boundary=decision_boundary,
            explainability_methods=cfg.explainers,
            batch_size=cfg.batch_size,
            cache_explanations=cfg.cache_explanations,
            save_path=results_dir,
        )


if __name__ == "__main__":
    main()

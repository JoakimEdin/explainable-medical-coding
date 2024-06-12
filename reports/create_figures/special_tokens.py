# ruff: noqa: E402
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import find_dotenv, load_dotenv

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
load_dotenv(find_dotenv())

import numpy as np
import polars as pl
import torch
from omegaconf import OmegaConf
from scipy.stats import pearsonr
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

from explainable_medical_coding.config.factories import get_explainability_method
from explainable_medical_coding.eval.plausibility_metrics import (
    calculate_plausibility_metrics,
)
from explainable_medical_coding.utils.analysis import get_explanations
from explainable_medical_coding.utils.data_helper_functions import (
    get_code2description_mimiciii,
)
from explainable_medical_coding.utils.loaders import (
    load_and_prepare_dataset,
    load_trained_model,
)
from explainable_medical_coding.utils.tokenizer import TargetTokenizer

pd.options.display.float_format = "{:,.2f}".format

pattern = "^[^A-Za-z0-9]*$"


def count_punctuation_tokens(attribution_df, note_id2tokens):
    def top5_special_tokens(attributions, note_id):
        tokens = note_id2tokens[note_id]
        top5tokens = tokens[np.argsort(attributions)[-5:]].tolist()
        return len([token for token in top5tokens if re.match(pattern, token)])

    tokens = attribution_df.select(
        pl.struct("attributions", "note_id")
        .map_elements(
            lambda row: top5_special_tokens(row["attributions"], row["note_id"])
        )
        .alias("special_tokens")
    )
    return tokens["special_tokens"].to_numpy().mean() / 5


def count_ground_truth_punctuation_tokens(attribution_df, note_id2tokens):
    def top5_special_tokens(evidence_token_ids, note_id):
        tokens = note_id2tokens[note_id]
        top5tokens = tokens[evidence_token_ids].tolist()
        return len([token for token in top5tokens if re.match(pattern, token)])

    tokens = attribution_df.select(
        pl.struct("evidence_token_ids", "note_id")
        .map_elements(
            lambda row: top5_special_tokens(row["evidence_token_ids"], row["note_id"])
        )
        .alias("special_tokens")
    )
    total_tokens = attribution_df["evidence_token_ids"].list.len().sum()
    totan_special_tokens = tokens["special_tokens"].sum()
    return totan_special_tokens / total_tokens


def set_special_token_attributions_to_zero(attribution_df, note_id2tokens):
    def set_special_tokens_to_zero(attributions, note_id):
        tokens = note_id2tokens[note_id]
        special_tokens = np.array([bool(re.match(pattern, token)) for token in tokens])
        attributions = np.array(attributions)
        attributions[special_tokens] = 0
        return list(attributions / (np.sum(attributions) + 1e-10))

    return attribution_df.with_columns(
        pl.struct("attributions", "note_id")
        .map_elements(
            lambda row: set_special_tokens_to_zero(row["attributions"], row["note_id"])
        )
        .alias("attributions")
    )

figure_path = Path("reports/figures/special_tokens")
figure_path.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_columns = ["diagnosis_codes", "procedure_codes"]
dataset_path = Path("explainable_medical_coding/datasets/mdace_inpatient_icd9.py")
note_id2tokens: dict[str, np.array] = {}
explainers_map = {
    "random": "Rand",
    "laat": "$a$",
    "attention_rollout": "Rollout",
    "deeplift": "Deeplift",
    "integrated_gradient": "IG",
    "gradient_x_input": "$x \\nabla x$",
    "grad_attention": "$a x \\nabla x$",
    "atgrad_attention": "$a \\nabla a$",
    "alti": "Alti",
    "occlusion": "Occl",
    "kernelshap": "SHAP",
    "lime": "LIME",
}
percentage_metrics = [
    "precision",
    "recall",
    "f1",
    "no_predictions",
    "average_number_of_predicted_tokens",
    "attributions_sum_zero",
    "evidence_span_token_recall",
    "evidence_span_recall",
    "auprc",
    "iou",
    "precision@1",
    "recall@1",
    "precision@5",
    "recall@5",
    "precision@10",
    "recall@10",
    "precision@50",
    "recall@50",
    "precision@100",
    "recall@100",
    "comprehensiveness",
    "sufficiency",
]
non_percentage_metrics = ["entropy", "kl_divergence", "decision_boundary"]
explanation_methods = ["laat", "gradient_x_input", "grad_attention"]
file_name_plausibility = "plausibility_and_sparsity.csv"
file_name_faithfulness = "faithfulness.csv"
results_dir = Path("reports/explainability_results")
model_dirs = {
    "B$_{\\text{Sup}}$": [results_dir / "supervised_sweep"],
    "B$_{\\text{Uns}}$": [
        results_dir / "unsupervised_sweep",
        results_dir / "unsupervised_more_sweep",
    ],
    "IGR": [results_dir / "igr_sweep"],
    "FM": [results_dir / "tm_sweep"],
    "PGD": [results_dir / "ant_sweep"],
}
suffix_map = {
    "B$_{\\text{Sup}}$": "_supervised",
    "B$_{\\text{Uns}}$": "_unsupervised",
    "IGR": "_igr",
    "FM": "_fm",
    "PGD": "_pgd",
}

dataframe_path = figure_path / "special_token_results.csv"
if dataframe_path.exists():
    df = pd.read_csv(dataframe_path)
else:
    dataframes = []
    for model_name, model_dir_list in model_dirs.items():
        model_dataframes = []
        for model_dir_name in model_dir_list:
            for subdir in model_dir_name.iterdir():
                if subdir.is_dir():
                    plausibility_path = subdir / file_name_plausibility
                    if not plausibility_path.exists():
                        continue

                    df_plaus = pd.read_csv(plausibility_path)
                    df_plaus["run_id"] = subdir.name

                    faithfulness_path = subdir / file_name_faithfulness
                    if faithfulness_path.exists():
                        df_faith = pd.read_csv(faithfulness_path)
                        df_plaus = pd.merge(
                            df_plaus,
                            df_faith,
                            on="explainability_method",
                            suffixes=("_plaus", "_faith"),
                        )

                    model_dataframes.append(df_plaus)

        df = pd.concat(model_dataframes)

        df[percentage_metrics] = df[percentage_metrics] * 100
        df["Model"] = model_name
        dataframes.append(df)

    df = pd.concat(dataframes)
    df = df.rename(columns={"explainability_method": "Explainer"})

    df = df[df["prediction_split"] == "all"]
    df = df[df["Explainer"].isin(explanation_methods)]
    df["special_token_rate"] = None

    model_dir = Path("models")
    run_id2subdir = {
        run_id: subdir
        for subdir in model_dir.iterdir()
        for run_id in subdir.iterdir()
    }

    for run_id in df["run_id"].unique():
        model_path = Path("models") / run_id2subdir[run_id] / run_id
        saved_config = OmegaConf.load(model_path / "config.yaml")
        text_tokenizer_path = saved_config.model.configs.model_path
        code2description = get_code2description_mimiciii()
        target_tokenizer = TargetTokenizer(autoregressive=False)
        target_tokenizer.load(model_path / "target_tokenizer.json")

        text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_path)
        max_input_length = int(6000)

        datasets = load_and_prepare_dataset(
            dataset_path,
            text_tokenizer,
            target_tokenizer,
            max_input_length,
            target_columns,
        )
        datasets = datasets.filter(
            lambda x: x["note_type"] == "Discharge summary",
            desc="Filtering all notes that are not discharge summaries",
        )

        model, decision_boundary = load_trained_model(
            model_path,
            saved_config,
            pad_token_id=text_tokenizer.pad_token_id,
            device=device,
        )
        for explainability_method in explanation_methods:
            explainer = get_explainability_method(explainability_method)
            explainer_callable = explainer(
                model=model,
                baseline_token_id=text_tokenizer.mask_token_id,
                cls_token_id=text_tokenizer.cls_token_id,
                eos_token_id=text_tokenizer.eos_token_id,
            )

            explanations_df = get_explanations(
                model=model,
                model_path=model_path,
                dataset=datasets["test"],
                explainer=explainer_callable,
                target_tokenizer=target_tokenizer,
                decision_boundary=decision_boundary,
                cache_path=Path(".cache"),
                cache=True,
                overwrite_cache=True,
            )
            explanations_df = explanations_df.filter(
                pl.col("evidence_token_ids").list.len() > 0
            )  # filter out empty groundtruth explanations. They are empty when the evidence is in the truncated text
            if len(note_id2tokens) == 0:
                note_ids = explanations_df["note_id"].unique().to_list()
                note_id2tokens = {
                    note_id: np.array(
                        text_tokenizer.convert_ids_to_tokens(
                            datasets["test"].filter(lambda x: x["note_id"] == note_id)[
                                0
                            ]["input_ids"]
                        )
                    )
                    for note_id in note_ids
                }

            explanations_no_special_df = set_special_token_attributions_to_zero(
                explanations_df, note_id2tokens
            )
            # remove start and end tokens from the attributions
            explanations_no_special_df = explanations_no_special_df.with_columns(
                pl.col("attributions").map_elements(lambda x: x[1:-1])
            )
            # shift evidence token ids by 1 to account for the removed start token
            explanations_no_special_df = explanations_no_special_df.with_columns(
                evidence_token_ids=pl.col("evidence_token_ids").map_elements(
                    lambda x: [i - 1 for i in x]
                )
            )
            explanations_no_special_df = explanations_no_special_df.with_columns(
                pl.col("attributions").map_elements(
                    lambda x: (np.array(x) / (sum(x) + 1e-11)).tolist()
                )
            )
            db = df.loc[
                (df["run_id"] == run_id) & (df["Explainer"] == explainability_method),
                "decision_boundary",
            ].values[0]
            results_dict = calculate_plausibility_metrics(
                explanations_no_special_df, db
            )

            for metric, value in results_dict.items():
                df.loc[
                    (df["run_id"] == run_id)
                    & (df["Explainer"] == explainability_method),
                    f"{metric}_no_special",
                ] = value * 100

            fraction_of_special_tokens = count_punctuation_tokens(
                explanations_df, note_id2tokens
            )
            ground_truth_fraction_of_special_tokens = (
                count_ground_truth_punctuation_tokens(explanations_df, note_id2tokens)
                * 100
            )
            df.loc[
                (df["run_id"] == run_id) & (df["Explainer"] == explainability_method),
                "special_token_rate",
            ] = fraction_of_special_tokens * 100
            df.loc[
                (df["run_id"] == run_id) & (df["Explainer"] == explainability_method),
                "ground_truth_special_token_rate",
            ] = ground_truth_fraction_of_special_tokens

    df["Explainer"] = df["Explainer"].map(explainers_map)
    df.to_csv(figure_path / "special_token_results.csv", index=False)


sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=2)
# make a custom list of colors
new_colors_order = sns.color_palette(n_colors=5)
new_colors_order[0], new_colors_order[1] = new_colors_order[1], new_colors_order[0]
# # df = df[df["Explainer"]=="laat"]

ground_truth_fraction_of_special_tokens = df["ground_truth_special_token_rate"].iloc[0]
# df = df[df["Explainer"].isin({"$a$", "$x \\nabla x$", "$a x \\nabla x$"})]
for model_name in df["Model"].unique():
    df_model = df[df["Model"] == model_name]
    print(model_name)
    print(
        df_model.groupby("Explainer")[
            [
                "special_token_rate",
                "f1",
                "iou",
                "recall@5",
                "precision@5",
                "auprc",
                "comprehensiveness",
                "sufficiency",
            ]
        ].corr()
    )
    print(
        df_model[df_model["Explainer"].isin(["$a x \\nabla x$", "$a$"])][
            [
                "special_token_rate",
                "f1",
                "iou",
                "recall@5",
                "precision@5",
                "auprc",
                "comprehensiveness",
                "sufficiency",
            ]
        ].corr()
    )
    suffix = suffix_map[model_name]

    def scatter_plot(metric):
        explainers = df_model["Explainer"].unique()
        g = sns.scatterplot(
            df_model,
            x="special_token_rate",
            y=metric,
            hue="Explainer",
            palette=new_colors_order,
            s=100,
        )
        x_plot = np.array([0, df_model["special_token_rate"].max() + 5])
        for idx, explainer in enumerate(explainers):
            df_explainer = df_model[df_model["Explainer"] == explainer]

            # regression line
            x = df_explainer["special_token_rate"].values
            y = df_explainer[metric].values

            # standardize
            x_scaler, y_scaler = StandardScaler(), StandardScaler()
            x_train = x_scaler.fit_transform(x[..., None])
            y_train = y_scaler.fit_transform(y[..., None])

            # fit model
            model = HuberRegressor(epsilon=2)
            model.fit(x_train, y_train.ravel())

            predictions = y_scaler.inverse_transform(
                model.predict(x_scaler.transform(x_plot[..., None]))[..., None]
            ).squeeze()
            # use same colors as in scatter plot
            color = new_colors_order[idx]
            plt.plot(x_plot, predictions, color=color, linestyle="-", linewidth=2)

            pearsonr_val = pearsonr(x, y)
            # plt.title(f"$r={pearsonr_val.statistic:.2f}$")
            # add pearson correlation in plot next to the regression line
            x_mid = x_plot[-1] / 2
            y_mid = y_scaler.inverse_transform(
                model.predict(x_scaler.transform([[x_mid]]))[..., None]
            ).squeeze()
            x_dif = x_plot[-1] - x_plot[0]
            y_dif = predictions[0] - predictions[-1]

            if metric == "comprehensiveness":
                if idx == 0:
                    y_mid = y_mid + y_dif / 3
                    x_mid = x_mid - x_dif / 10

                elif idx == 1:
                    y_mid = y_mid - 5
                    x_mid = x_mid - 10
                else:
                    y_mid = y_mid - y_dif / 2

            else:
                if idx == 0:
                    y_mid = y_mid - y_dif / 2
                elif idx == 1:
                    y_mid = y_mid - y_dif / 3
                    x_mid = x_mid - x_dif / 10
                else:
                    y_mid = y_mid + y_dif / 3
                    x_mid = x_mid - x_dif / 10

            plt.text(
                x_mid,
                y_mid,
                f"$r={pearsonr_val.statistic:.2f}$",
                color=color,
                fontsize=20,
            )

        plt.xlim(x_plot)
        plt.xlabel("Special Tokens in Top 5 (%)")
        _, xlabels = plt.xticks()
        g.set_xticklabels(xlabels, size=20)
        plt.axvline(
            x=ground_truth_fraction_of_special_tokens,
            color="black",
            linestyle="--",
            alpha=0.4,
        )
        plt.tight_layout()
        g.legend_.set_title(None)

    scatter_plot("f1")
    plt.ylabel("F1 Score (%)")
    plt.savefig(figure_path / f"scatter_f1{suffix}.pdf", format="pdf")
    plt.clf()

    scatter_plot("iou")
    plt.ylabel("IOU (%)")
    plt.savefig(figure_path / f"scatter_iou{suffix}.pdf", format="pdf")
    plt.clf()

    scatter_plot("recall@5")
    plt.ylabel("R@5 (%)")
    plt.savefig(figure_path / f"scatter_r5{suffix}.pdf", format="pdf")
    plt.clf()

    scatter_plot("precision@5")
    plt.ylabel("P@5 (%)")
    plt.savefig(figure_path / f"scatter_p5{suffix}.pdf", format="pdf")
    plt.clf()

    scatter_plot("auprc")
    plt.ylabel("AUPRC (%)")
    plt.savefig(figure_path / f"scatter_auprc{suffix}.pdf", format="pdf")
    plt.clf()

    scatter_plot("comprehensiveness")
    plt.ylabel("Comprehensiveness (%)")
    plt.savefig(figure_path / f"scatter_comprehensiveness{suffix}.pdf", format="pdf")
    plt.clf()

    scatter_plot("sufficiency")
    plt.ylabel("Sufficiency (%)")
    plt.savefig(figure_path / f"scatter_sufficiency{suffix}.pdf", format="pdf")
    plt.clf()

from pathlib import Path

import numpy as np
import pandas as pd

pd.options.display.float_format = "{:,.2f}".format

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
]
non_percentage_metrics = ["entropy", "kl_divergence", "decision_boundary"]

file_name = "plausibility_and_sparsity.csv"
results_dir = Path("reports/explainability_results")
model_dirs = {
    # "Supervised": "reports/explainability_results/supervised_ag/plausibility_and_sparsity.csv",
    "B$_{\\text{U}}$": results_dir / "unsupervised_full_sweep",
    "IGR": results_dir / "igr_full_sweep",
    "PGD": results_dir / "pgd_full_sweep",
    "TM": results_dir / "tm_full_sweep",
}
explainer_order = [
    "Rand",
    "$a$",
    "Rollout",
    "Alti",
    "$a \\nabla a$",
    "$a x \\nabla x$",
    "$x \\nabla x$",
    "IG",
    "Deeplift",
    "Occl",
    "SHAP",
    "LIME",
]
model_order = ["B$_{\\text{U}}$", "PGD", "TM", "IGR"]
dataframes = []
model_name = "B$_{\\text{U}}$"
subdirs = [
    results_dir / "unsupervised_sweep" / "1k5b5cxg",
    results_dir / "perturbation2" / "1k5b5cxg",
]

model_dataframes = []
for subdir in subdirs:
    if subdir.is_dir():
        path = subdir / file_name
        if not path.exists():
            continue
        model_dataframes.append(pd.read_csv(path))

df = pd.concat(model_dataframes)

df[percentage_metrics] = df[percentage_metrics] * 100

df_percentage = (
    df.groupby(["explainability_method", "prediction_split"])[percentage_metrics]
    .agg(lambda x: f"{np.mean(x):.1f}$\\pm${np.std(x):.1f}")
    .reset_index()
)
df_non_percentage = (
    df.groupby(["explainability_method", "prediction_split"])[non_percentage_metrics]
    .agg(lambda x: f"{np.mean(x):.2f}$\\pm${np.std(x):.2f}")
    .reset_index()
)
df = pd.merge(
    df_percentage,
    df_non_percentage,
    on=["explainability_method", "prediction_split"],
)

df["Model"] = model_name

dataframes.append(df)


plausibility = pd.concat(dataframes).reset_index(drop=True)


plausibility = plausibility.rename(
    columns={
        "explainability_method": "Explainer",
        "prediction_split": "Pred",
        "average_number_of_predicted_tokens": "# predicted tokens",
        "evidence_span_token_recall": "Cover",
        "evidence_span_recall": "Recall",
        "entropy": "Entropy",
        "no_predictions": "Empty",
        "f1": "F1",
        "auprc": "AUPRC",
        "iou": "IOU",
    }
)
plausibility["Explainer"] = plausibility["Explainer"].apply(lambda x: explainers_map[x])
plausibility.columns = (
    plausibility.columns.str.replace("_", " ")
    .str.replace("precision", "P")
    .str.replace("recall", "R")
)

plausibility["Pred"] = plausibility["Pred"].replace(
    {"all": "both", "predicted": "tp", "not_predicted": "fn"}
)


# sort rows according to explainer_order
plausibility["Explainer"] = pd.Categorical(
    plausibility["Explainer"], categories=explainer_order, ordered=True
)
plausibility = plausibility.sort_values("Explainer")

plausibility = plausibility[
    [
        "Explainer",
        "Pred",
        "P",
        "R",
        "F1",
        "AUPRC",
        "Empty",
        "Cover",
        "Recall",
        "IOU",
        "P@5",
        "R@5",
    ]
]
plausibility.columns = pd.MultiIndex.from_tuples(
    [
        ("", "Explainer"),
        ("", "Pred"),
        ("Prediction", "P $\\uparrow$"),
        ("Prediction", "R $\\uparrow$"),
        ("Prediction", "F1 $\\uparrow$"),
        ("Prediction", "AUPRC $\\uparrow$"),
        ("Prediction", "Empty $\\downarrow$"),
        ("Prediction", "SpanR $\\uparrow$"),
        ("Prediction", "Cover $\\uparrow$"),
        ("Ranking", "IOU $\\uparrow$"),
        ("Ranking", "P@5 $\\uparrow$"),
        ("Ranking", "R@5 $\\uparrow$"),
    ]
)
# sort according to explainer order


# print tp only
plausibility_tp = plausibility[plausibility[("", "Pred")] == "tp"].drop(
    columns=[("", "Pred")]
)
# hide index
plausibility_tp = plausibility_tp.style.format(
    decimal=".", thousands=" ", precision=2
).hide(axis="index")
print(
    plausibility_tp.to_latex(
        hrules=True,
        position_float="centering",
        multicol_align="c",
        label="tab:plausibility_tp",
    )
)

# print fn only
plausibility_fn = plausibility[plausibility[("", "Pred")] == "fn"].drop(
    columns=[("", "Pred")]
)
plausibility_fn = plausibility_fn.style.format(
    decimal=".",
    thousands=" ",
    precision=2,
).hide(axis="index")
print(
    plausibility_fn.to_latex(
        hrules=True,
        position_float="centering",
        multicol_align="c",
        label="tab:plausibility_fn",
        column_format="ll" + "c" * len(plausibility.columns),
    )
)

# print fn only
plausibility_both = plausibility[plausibility[("", "Pred")] == "both"].drop(
    columns=[("", "Pred")]
)
plausibility_both = plausibility_both.style.format(
    decimal=".",
    thousands=" ",
    precision=2,
).hide(axis="index")
# add midrule after each model
print(
    plausibility_both.to_latex(
        hrules=True,
        position_float="centering",
        multicol_align="c",
        label="tab:plausibility_both",
        column_format="ll" + "c" * len(plausibility.columns),
    )
)

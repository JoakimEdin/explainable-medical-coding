from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pd.options.display.float_format = "{:,.2f}".format
figure_path = Path("reports/figures/plausibility")
figure_path.mkdir(exist_ok=True, parents=True)

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
supervised = True
if supervised:
    model_dirs = {
        # "Supervised": "reports/explainability_results/supervised_ag/plausibility_and_sparsity.csv",
        "B$_{\\text{S}}$": results_dir / "supervised_sweep",
        "B$_{\\text{U}}$": results_dir / "unsupervised_sweep",
        "IGR": results_dir / "igr_sweep",
        "TM": results_dir / "tm_sweep",
        "PGD": results_dir / "ant_sweep",
    }
    suffix = "_supervised"
else:
    model_dirs = {
        # "Supervised": "reports/explainability_results/supervised_ag/plausibility_and_sparsity.csv",
        "B$_{\\text{U}}$": results_dir / "unsupervised_full_sweep",
        "TM": results_dir / "tm_full_sweep",
        "IGR": results_dir / "igr_full_sweep",
    }
    suffix = "_unsupervised"

model_order = ["B$_{\\text{U}}$", "B$_{\\text{S}}$", "IGR", "TM", "PGD"]
dataframes = []
for model_name, model_dir_name in model_dirs.items():
    model_dataframes = []
    for subdir in model_dir_name.iterdir():
        if subdir.is_dir():
            path = subdir / file_name
            if not path.exists():
                continue
            model_dataframes.append(pd.read_csv(path))

    df = pd.concat(model_dataframes)

    df[percentage_metrics] = df[percentage_metrics] * 100
    df["Model"] = model_name
    df["run_id"] = subdir.name
    dataframes.append(df)

df = pd.concat(dataframes)
df = df.rename(columns={"explainability_method": "Explainer"})
df["Explainer"] = df["Explainer"].map(explainers_map)
df = df[df["prediction_split"] == "all"]
# df = df[["Explainer","Model", "run_id"]]
# add a column called metrics and one called value. Transform the dataframe to long format
df = df.melt(
    id_vars=["Explainer", "Model", "run_id"],
    value_vars=percentage_metrics + non_percentage_metrics,
    var_name="metric",
    value_name="value",
)
sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=2)
# df = df[df["Explainer"]=="laat"]

df = df[df["Explainer"].isin({"$a$", "$x \\nabla x$", "$a x \\nabla x$"})]
g = sns.boxplot(df[df["metric"] == "f1"], x="Explainer", y="value", hue="Model")
plt.ylabel("F1 Score (%)")
plt.xlabel("")
_, xlabels = plt.xticks()
g.set_xticklabels(xlabels, size=16)
plt.savefig(figure_path / f"boxplot_f1{suffix}.pdf", format="pdf")
plt.clf()


g = sns.boxplot(df[df["metric"] == "iou"], x="Explainer", y="value", hue="Model")
plt.ylabel("Intersection over Unions (%)")
plt.xlabel("")
_, xlabels = plt.xticks()
g.set_xticklabels(xlabels, size=16)
plt.savefig(figure_path / f"boxplot_iou{suffix}.pdf", format="pdf")
plt.clf()

g = sns.boxplot(df[df["metric"] == "auprc"], x="Explainer", y="value", hue="Model")
plt.ylabel("Area Under the Precision-Recall Curve (%)")
plt.xlabel("")
_, xlabels = plt.xticks()
g.set_xticklabels(xlabels, size=16)
plt.savefig(figure_path / f"boxplot_auprc{suffix}.pdf", format="pdf")
plt.clf()

if supervised:
    df["method"] = ""
    df.loc[
        df[["Explainer", "Model"]].isin(["$a$", "B$_{\\text{U}}$"]).all(axis=1),
        "method",
    ] = "$a$+B$_{\\text{U}}$"
    df.loc[
        df[["Explainer", "Model"]].isin(["$a$", "B$_{\\text{S}}$"]).all(axis=1),
        "method",
    ] = "$a$+B$_{\\text{S}}$"
    df.loc[
        df[["Explainer", "Model"]].isin(["$a x \\nabla x$", "TM"]).all(axis=1), "method"
    ] = "$a x \\nabla x$+TM"
    df = df[
        df["method"].isin(
            ["$a$+B$_{\\text{S}}$", "$a$+B$_{\\text{U}}$", "$a x \\nabla x$+TM"]
        )
    ]
    df["metric"] = df["metric"].str.upper().str.replace("NO_PREDICTIONS", "Empty")
    df["method"] = pd.Categorical(
        df["method"],
        categories=[
            "$a$+B$_{\\text{S}}$",
            "$a$+B$_{\\text{U}}$",
            "$a x \\nabla x$+TM",
        ],
        ordered=True,
    )
    df["annotations"] = "Unsupervised"
    df.loc[df["Model"] == "B$_{\\text{S}}$", "annotations"] = "Supervised"

    g = sns.boxplot(
        df[df["metric"].isin(["F1"])], x="method", y="value", hue="annotations"
    )
    plt.ylabel("F1 Score (%)")
    plt.xlabel("")
    # get label text
    _, xlabels = plt.xticks()
    # set the x-labels with
    g.set_xticklabels(xlabels, size=20)
    g.legend_.set_title(None)
    plt.savefig(figure_path / "boxplot_plausibility_comparison_f1.pdf", format="pdf")
    plt.clf()

    g = sns.boxplot(
        df[df["metric"].isin(["IOU"])], x="method", y="value", hue="annotations"
    )
    plt.ylabel("IOU (%)")
    plt.xlabel("")
    # get label text
    _, xlabels = plt.xticks()
    # set the x-labels with
    g.set_xticklabels(xlabels, size=20)
    g.legend_.set_title(None)
    plt.savefig(figure_path / "boxplot_plausibility_comparison_iou.pdf", format="pdf")
    plt.clf()

    g = sns.boxplot(
        df[df["metric"].isin(["Empty"])], x="method", y="value", hue="annotations"
    )
    plt.ylabel("Empty (%)")
    plt.xlabel("")
    g.legend_.set_title(None)
    # get label text
    _, xlabels = plt.xticks()
    # set the x-labels with
    g.set_xticklabels(xlabels, size=20)
    plt.savefig(figure_path / "boxplot_plausibility_comparison_empty.pdf", format="pdf")
    plt.clf()

    g = sns.boxplot(
        df[df["metric"].isin(["RECALL@5"])], x="method", y="value", hue="annotations"
    )
    plt.ylabel("Recall@5 (%)")
    plt.xlabel("")
    # get label text
    _, xlabels = plt.xticks()
    # set the x-labels with
    g.set_xticklabels(xlabels, size=20)
    g.legend_.set_title(None)
    plt.savefig(figure_path / "boxplot_plausibility_comparison_r5.pdf", format="pdf")
    plt.clf()

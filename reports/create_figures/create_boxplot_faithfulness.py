import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


pd.options.display.float_format = "{:,.2f}".format
figure_path = Path("reports/figures/faithfulness")
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

file_name = "faithfulness.csv"
results_dir = Path("reports/explainability_results")
supervised = True
if supervised:
    model_dirs = {
        # "Supervised": "reports/explainability_results/supervised_ag/plausibility_and_sparsity.csv",
        "B$_{\\text{S}}$": results_dir / "supervised_sweep",
        "B$_{\\text{U}}$": results_dir / "unsupervised_sweep",
        "IGR": results_dir / "igr_sweep",
        "TM": results_dir / "tm_sweep",
        "PGD": results_dir / "pgd_sweep",
    }
    suffix = "_supervised"
else:
    model_dirs = {
        # "Supervised": "reports/explainability_results/supervised_ag/plausibility_and_sparsity.csv",
        "B$_{\\text{U}}$": results_dir / "unsupervised_full_sweep",
        "IGR": results_dir / "igr_full_sweep",
        "TM": results_dir / "tm_full_sweep",
        "PGD": results_dir / "pgd_full_sweep",
    }
    suffix = "_unsupervised"

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

    df["Model"] = model_name
    df["run_id"] = subdir.name
    dataframes.append(df)

df = pd.concat(dataframes)
df = df.rename(columns={"explainability_method": "Explainer"})
df["Explainer"] = df["Explainer"].map(explainers_map)
# df = df[["Explainer","Model", "run_id"]]
# add a column called metrics and one called value. Transform the dataframe to long format
df = df.melt(
    id_vars=["Explainer", "Model", "run_id"],
    value_vars=["comprehensiveness", "sufficiency"],
    var_name="metric",
    value_name="value",
)
sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=2)
# make a custom list of colors
new_colors_order = sns.color_palette(n_colors=5)
new_colors_order[0], new_colors_order[1] = new_colors_order[1], new_colors_order[0]
# df = df[df["Explainer"]=="laat"]
df = df[df["Explainer"].isin({"$a$", "$x \\nabla x$", "$a x \\nabla x$"})]
g = sns.boxplot(
    df[df["metric"] == "comprehensiveness"],
    x="Model",
    y="value",
    hue="Explainer",
    palette=new_colors_order,
)
g.get_legend().set_title("")
plt.ylabel("Comprehensiveness")
plt.xlabel("")
plt.ylim([0.4, 0.91])
_, xlabels = plt.xticks()
g.set_xticklabels(xlabels, size=20)
plt.savefig(figure_path / f"boxplot_comprehensiveness{suffix}.pdf", format="pdf")
plt.clf()

g = sns.boxplot(
    df[df["metric"] == "sufficiency"],
    x="Model",
    y="value",
    hue="Explainer",
    palette=new_colors_order,
)
g.get_legend().set_title("")
plt.ylabel("Sufficiency")
plt.xlabel("")
_, xlabels = plt.xticks()
g.set_xticklabels(xlabels, size=20)
plt.savefig(figure_path / f"boxplot_sufficiency{suffix}.pdf", format="pdf")

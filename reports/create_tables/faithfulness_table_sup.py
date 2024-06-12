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
file_name = "faithfulness.csv"
results_dir = Path("reports/explainability_results")
model_dirs = {
    "B$_{\\text{U}}$": results_dir / "unsupervised_sweep",
    "B$_{\\text{S}}$": results_dir / "supervised_sweep",
    "IGR": results_dir / "igr_sweep",
    "TM": results_dir / "tm_sweep",
    "PGD": results_dir / "pgd_sweep",
}
explainer_order = [
    "Rand",
    "$a$",
    "Occl",
    "SHAP",
    "LIME",
    "$x \\nabla x$",
    "IG",
    "Deeplift",
    "Rollout",
    "Alti",
    "$a \\nabla a$",
    "$a x \\nabla x$",
]
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
    df = (
        df.groupby(["explainability_method"])
        .agg(lambda x: f"{np.mean(x):.2f}$\\pm${np.std(x):.2f}")
        .reset_index()
    )
    df["Model"] = model_name
    dataframes.append(df)


faithfulness = pd.concat(dataframes).reset_index(drop=True)


faithfulness = faithfulness.rename(columns={"explainability_method": "Explainer"})
faithfulness["Explainer"] = faithfulness["Explainer"].apply(lambda x: explainers_map[x])
faithfulness["Explainer"] = pd.Categorical(
    faithfulness["Explainer"], categories=explainer_order, ordered=True
)
faithfulness["Model"] = pd.Categorical(
    faithfulness["Model"], categories=model_order, ordered=True
)
faithfulness = faithfulness.sort_values(["Model", "Explainer"])


index_tuples = [
    (model, explainer)
    for explainer, model in zip(faithfulness["Explainer"], faithfulness["Model"])
]
faithfulness.index = pd.MultiIndex.from_tuples(
    index_tuples, names=["Model", "Explainer"]
)
faithfulness = faithfulness.drop(columns=["Model", "Explainer"])

faithfulness.columns = faithfulness.columns.str.title()
faithfulness = faithfulness.rename(
    columns={
        "Comprehensiveness": "Comprehensiveness ($\\uparrow$)",
        "Sufficiency": "Sufficiency ($\\downarrow$)",
    }
)
faithfulness = faithfulness.style.format(
    decimal=".",
    thousands=" ",
    precision=2,
)
print(faithfulness.to_latex(hrules=True, multicol_align=True))

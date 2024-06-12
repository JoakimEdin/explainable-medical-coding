"""
Script to create the mimiciii_full dataset from the mimiciii dataset. The dataset is from the Mullenbach et al. paper.
"""

from pathlib import Path

import polars as pl


def convert_to_older_icd9_version(
    old_codes: set[str], new_codes: set[str]
) -> dict[str, str]:
    """Converts the old ICD-9 codes to the new ICD-9 codes.

    Args:
        old_codes (set[str]): The old ICD-9 codes.
        new_codes (set[str]): The new ICD-9 codes.

    Returns:
        dict[str, str]: The mapping from old ICD-9 codes to new ICD-9 codes.
    """
    mapping = {}

    for new_code in new_codes:
        if new_code in old_codes:
            mapping[new_code] = new_code
        else:
            temp_new_code = new_code[:-1]
            while True:
                if len(temp_new_code) < 3:
                    print(f"Could not find mapping for {new_code}")
                    break
                if temp_new_code in old_codes:
                    print(f"Found mapping from {new_code} to {temp_new_code}")
                    mapping[new_code] = temp_new_code
                    break
                temp_new_code = temp_new_code[:-1]

    return mapping


new_folder_path = Path("data/processed/mdace_icd9_inpatient_code")
# create folder
new_folder_path.mkdir(parents=True, exist_ok=True)

notes = pl.read_parquet("data/processed/mdace_notes.parquet")

mimiciii = pl.read_parquet("data/processed/mimiciii.parquet")
mimiciii = mimiciii.unique(subset=["_id"])

annotations = pl.read_parquet("data/processed/mdace_inpatient_annotations.parquet")
annotations_icd9 = annotations.filter(pl.col("code_type").is_in({"icd9cm", "icd9pcs"}))

mimiciii_icd9cm_codes = set(
    mimiciii.filter(pl.col("diagnosis_code_type") == "icd9cm")["diagnosis_codes"]
    .explode()
    .unique()
)
mimiciii_icd9pcs_codes = set(
    mimiciii.filter(pl.col("procedure_code_type") == "icd9pcs")["procedure_codes"]
    .explode()
    .unique()
)

annotations_icd9cm_codes = set(
    annotations_icd9.filter(pl.col("code_type") == "icd9cm")["code"].explode().unique()
)
annotations_icd9pcs_codes = set(
    annotations_icd9.filter(pl.col("code_type") == "icd9pcs")["code"].explode().unique()
)

icd9cm_mapping = convert_to_older_icd9_version(
    mimiciii_icd9cm_codes, annotations_icd9cm_codes
)
icd9pcs_mapping = convert_to_older_icd9_version(
    mimiciii_icd9pcs_codes, annotations_icd9pcs_codes
)
annotations_icd9 = annotations_icd9.with_columns(
    pl.col("code")
    .map_elements(lambda code: icd9cm_mapping.get(code, code))
    .alias("code")
)
annotations_icd9 = annotations_icd9.with_columns(
    pl.col("code")
    .map_elements(lambda code: icd9pcs_mapping.get(code, code))
    .alias("code")
)

annotations_icd9_cm = (
    annotations_icd9.filter(pl.col("code_type") == "icd9cm")
    .group_by(["note_id"])
    .agg(
        pl.col("code").map_elements(list).alias("diagnosis_codes"),
        pl.col("spans").map_elements(list).alias("diagnosis_code_spans"),
        pl.col("code_type").last().alias("diagnosis_code_type"),
    )
)
annotations_icd9_pcs = (
    annotations_icd9.filter(pl.col("code_type") == "icd9pcs")
    .group_by(["note_id"])
    .agg(
        pl.col("code").map_elements(list).alias("procedure_codes"),
        pl.col("spans").map_elements(list).alias("procedure_code_spans"),
        pl.col("code_type").last().alias("procedure_code_type"),
    )
)

annotations_icd9 = annotations_icd9_cm.join(
    annotations_icd9_pcs, on="note_id", how="outer_coalesce"
)
mdace = notes.join(annotations_icd9, on="note_id")
mdace = mdace.drop(["CHARTDATE", "CHARTTIME", "STORETIME", "CGID", "ISERROR"])

mimiciii = mimiciii.filter(~pl.col("_id").is_in(mdace["_id"]))
mimiciii = mimiciii.with_columns(diagnosis_code_spans=None, procedure_code_spans=None)
mimiciii = mimiciii.select(mdace.columns)  # reorder columns
data = pl.concat([mdace, mimiciii])

data = data.with_columns(
    [
        pl.col("diagnosis_codes").fill_null([]),
        pl.col("procedure_codes").fill_null([]),
        pl.col("diagnosis_code_spans").fill_null([[[]]]),
        pl.col("procedure_code_spans").fill_null([[[]]]),
        pl.col("diagnosis_code_type").fill_null("icd9cm"),
        pl.col("procedure_code_type").fill_null("icd9pcs"),
    ]
)

train_split = pl.read_csv(
    "data/splits/mdace/inpatient/MDace-code-ev-train.csv",
    has_header=False,
    new_columns=["_id"],
)
val_split = pl.read_csv(
    "data/splits/mdace/inpatient/MDace-code-ev-val.csv",
    has_header=False,
    new_columns=["_id"],
)
test_split = pl.read_csv(
    "data/splits/mdace/inpatient/MDace-code-ev-test.csv",
    has_header=False,
    new_columns=["_id"],
)

train = data.filter(pl.col("_id").is_in(train_split["_id"]))
val = data.filter(pl.col("_id").is_in(val_split["_id"]))
test = data.filter(pl.col("_id").is_in(test_split["_id"]))

train.write_parquet(new_folder_path / "train.parquet")
val.write_parquet(new_folder_path / "val.parquet")
test.write_parquet(new_folder_path / "test.parquet")

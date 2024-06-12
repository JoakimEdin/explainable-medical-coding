"""
Script to create the MCACE dataset from the mimiciii dataset. The dataset is from the Mullenbach et al. paper.
"""

from pathlib import Path

import polars as pl

new_folder_path = Path("data/processed/mdace_icd10_profee")
# create folder
new_folder_path.mkdir(parents=True, exist_ok=True)


def convert_to_older_icd10_version(
    old_codes: set[str], new_codes: set[str]
) -> dict[str, str]:
    """Converts the old ICD-10 codes to the new ICD-10 codes.

    Args:
        old_codes (set[str]): The old ICD-10 codes.
        new_codes (set[str]): The new ICD-10 codes.

    Returns:
        dict[str, str]: The mapping from old ICD-10 codes to new ICD-10 codes.
    """
    mapping = {}
    # diagnosis codes
    mapping["F32.A"] = "F32.9"
    mapping["S02.122A"] = "S02.19XA"
    mapping["D75.839"] = "D75.89"
    mapping["F10.139"] = "F10.10"
    mapping["G90.51"] = "G90.519"
    mapping["M79.60"] = "M79.609"
    mapping["I50.3"] = "I50.30"

    for new_code in new_codes:
        if new_code in mapping:
            continue

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


notes = pl.read_parquet("data/processed/mdace_notes.parquet")
mimiciv = pl.read_parquet("data/processed/mimiciv.parquet")
annotations = pl.read_parquet("data/processed/mdace_profee_annotations.parquet")
annotations_icd10 = annotations.filter(pl.col("code_type").is_in({"icd10cm"}))

mimiciv_icd10cm_codes = set(
    mimiciv.filter(pl.col("diagnosis_code_type") == "icd10cm")["diagnosis_codes"]
    .explode()
    .unique()
)

annotations_icd10cm_codes = set(
    annotations_icd10.filter(pl.col("code_type") == "icd10cm")["code"]
    .explode()
    .unique()
)

icd10cm_mapping = convert_to_older_icd10_version(
    mimiciv_icd10cm_codes, annotations_icd10cm_codes
)

annotations_icd10 = annotations_icd10.with_columns(
    pl.col("code")
    .map_elements(lambda code: icd10cm_mapping.get(code, code))
    .alias("code")
)

annotations_icd10 = (
    annotations_icd10.filter(pl.col("code_type") == "icd10cm")
    .group_by(["note_id"])
    .agg(
        pl.col("code").map_elements(list).alias("diagnosis_codes"),
        pl.col("spans").map_elements(list).alias("diagnosis_code_spans"),
        pl.col("code_type").last().alias("diagnosis_code_type"),
    )
)


data = notes.join(annotations_icd10, on="note_id")
data = data.drop(["CHARTDATE", "CHARTTIME", "STORETIME", "CGID", "ISERROR"])

train = data
val = data
test = data

train.write_parquet(new_folder_path / "train.parquet")
val.write_parquet(new_folder_path / "val.parquet")
test.write_parquet(new_folder_path / "test.parquet")

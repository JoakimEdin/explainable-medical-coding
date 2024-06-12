"""
Script to create the mimiciii_full dataset from the mimiciii dataset. The dataset is from the Mullenbach et al. paper.
"""

from pathlib import Path

import polars as pl

new_folder_path = Path("data/processed/mdace_icd10_inpatient")
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


def identify_duplicated_codes(input_list: list[str]) -> list[int]:
    """takes a list of string as inputs and returns a list of indices of all duplicated codes.

    Args:
        l (list[str]): a list of string

    Returns:
        list[int]: a list of indices of duplicated codes

    """
    duplicated_codes = []
    for i in range(len(input_list)):
        for j in range(i + 1, len(input_list)):
            if input_list[i] == input_list[j]:
                duplicated_codes.append(i)
                duplicated_codes.append(j)
    return duplicated_codes


def remove_duplicated_codes(x: tuple):
    note_id, codes, spans, code_type = x
    if len(codes) == len(set(codes)):
        return x

    duplicated_indices = identify_duplicated_codes(codes)

    new_span = [span for idx in duplicated_indices for span in spans[idx]]

    new_codes = []
    new_spans = []

    for idx in range(len(codes)):
        if idx not in duplicated_indices:
            new_codes.append(codes[idx])
            new_spans.append(spans[idx])

        if idx == duplicated_indices[0]:
            new_codes.append(codes[idx])
            new_spans.append(new_span)

    return note_id, new_codes, new_spans, code_type


notes = pl.read_parquet("data/processed/mdace_notes.parquet")
mimiciv = pl.read_parquet("data/processed/mimiciv.parquet")
annotations = pl.read_parquet("data/processed/mdace_inpatient_annotations.parquet")
annotations_icd10 = annotations.filter(
    pl.col("code_type").is_in({"icd10cm", "icd10pcs"})
)

mimiciv_icd10cm_codes = set(
    mimiciv.filter(pl.col("diagnosis_code_type") == "icd10cm")["diagnosis_codes"]
    .explode()
    .unique()
)
mimiciv_icd10pcs_codes = set(
    mimiciv.filter(pl.col("procedure_code_type") == "icd10pcs")["procedure_codes"]
    .explode()
    .unique()
)

annotations_icd10cm_codes = set(
    annotations_icd10.filter(pl.col("code_type") == "icd10cm")["code"]
    .explode()
    .unique()
)
annotations_icd10pcs_codes = set(
    annotations_icd10.filter(pl.col("code_type") == "icd10pcs")["code"]
    .explode()
    .unique()
)

icd10cm_mapping = convert_to_older_icd10_version(
    mimiciv_icd10cm_codes, annotations_icd10cm_codes
)
icd10pcs_mapping = convert_to_older_icd10_version(
    mimiciv_icd10pcs_codes, annotations_icd10pcs_codes
)
annotations_icd10 = annotations_icd10.with_columns(
    pl.col("code")
    .map_elements(lambda code: icd10cm_mapping.get(code, code))
    .alias("code")
)
annotations_icd10 = annotations_icd10.with_columns(
    pl.col("code")
    .map_elements(lambda code: icd10pcs_mapping.get(code, code))
    .alias("code")
)

annotations_icd10_cm = (
    annotations_icd10.filter(pl.col("code_type") == "icd10cm")
    .group_by(["note_id"])
    .agg(
        pl.col("code").map_elements(list).alias("diagnosis_codes"),
        pl.col("spans").map_elements(list).alias("diagnosis_code_spans"),
        pl.col("code_type").last().alias("diagnosis_code_type"),
    )
)
annotations_icd10_pcs = (
    annotations_icd10.filter(pl.col("code_type") == "icd10pcs")
    .group_by(["note_id"])
    .agg(
        pl.col("code").map_elements(list).alias("procedure_codes"),
        pl.col("spans").map_elements(list).alias("procedure_code_spans"),
        pl.col("code_type").last().alias("procedure_code_type"),
    )
)
column_names = annotations_icd10_cm.columns
annotations_icd10_cm = annotations_icd10_cm.map_rows(remove_duplicated_codes)
annotations_icd10_cm.columns = column_names

annotations_icd10 = annotations_icd10_cm.join(
    annotations_icd10_pcs, on="note_id", how="outer_coalesce"
)


data = notes.join(annotations_icd10, on="note_id")
data = data.drop(["CHARTDATE", "CHARTTIME", "STORETIME", "CGID", "ISERROR"])
data = data.with_columns(
    [
        pl.col("diagnosis_codes").fill_null([]),
        pl.col("procedure_codes").fill_null([]),
        pl.col("diagnosis_code_spans").fill_null([[[]]]),
        pl.col("procedure_code_spans").fill_null([[[]]]),
        pl.col("diagnosis_code_type").fill_null("icd10cm"),
        pl.col("procedure_code_type").fill_null("icd10pcs"),
    ]
)
train_split = pl.read_csv(
    "data/splits/mdace/inpatient/MDace-ev-train.csv",
    has_header=False,
    new_columns=["_id"],
)
val_split = pl.read_csv(
    "data/splits/mdace/inpatient/MDace-ev-val.csv",
    has_header=False,
    new_columns=["_id"],
)
test_split = pl.read_csv(
    "data/splits/mdace/inpatient/MDace-ev-test.csv",
    has_header=False,
    new_columns=["_id"],
)

train = data.filter(pl.col("_id").is_in(train_split["_id"]))
val = data.filter(pl.col("_id").is_in(val_split["_id"]))
test = data.filter(pl.col("_id").is_in(test_split["_id"]))

train.write_parquet(new_folder_path / "train.parquet")
val.write_parquet(new_folder_path / "val.parquet")
test.write_parquet(new_folder_path / "test.parquet")

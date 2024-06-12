"""
Script to create the mimiciii_clean dataset from the mimiciii dataset.
"""

from pathlib import Path

import polars as pl

from explainable_medical_coding.utils.data_helper_functions import remove_rare_codes

new_folder_path = Path("data/processed/mimiciii_clean")
# create folder
new_folder_path.mkdir(parents=True, exist_ok=True)

data = pl.read_parquet("data/processed/mimiciii.parquet")
splits = pl.read_ipc("data/splits/mimiciii_clean_splits.feather")
data = data.join(splits, on="_id")

# remove not used columns
data = data.drop(["note_seq", "charttime", "storetime"])

# only keep ICD-10 codes
data = data.filter(
    (pl.col("diagnosis_code_type") == "icd9cm")
    | (pl.col("procedure_code_type") == "icd9pcs")
)

# remove rare codes
data = remove_rare_codes(data, ["diagnosis_codes", "procedure_codes"], 10)

# drop duplicates
data = data.unique(subset=["_id"])

train = data.filter(pl.col("split") == "train")
val = data.filter(pl.col("split") == "val")
test = data.filter(pl.col("split") == "test")

# remove split column
train = train.drop("split")
val = val.drop("split")
test = test.drop("split")

# save files as parquet
train.write_parquet(new_folder_path / "train.parquet")
val.write_parquet(new_folder_path / "val.parquet")
test.write_parquet(new_folder_path / "test.parquet")

"""
Script to create the mimiciii_50 dataset from the mimiciii dataset. The dataset is from the Mullenbach et al. paper.
"""

from pathlib import Path

import polars as pl

from explainable_medical_coding.utils.data_helper_functions import keep_top_k_codes

new_folder_path = Path("data/processed/mimiciii_50")
# create folder
new_folder_path.mkdir(parents=True, exist_ok=True)

data = pl.read_parquet("data/processed/mimiciii.parquet")
splits = pl.read_ipc("data/splits/mimiciii_50_splits.feather")
data = data.join(splits, on="_id")

# remove not used columns
data = data.drop(["note_seq", "charttime", "storetime"])

# only keep ICD-10 codes
data = data.filter(
    (pl.col("diagnosis_code_type") == "icd9cm")
    | (pl.col("procedure_code_type") == "icd9pcs")
)

# drop duplicates
data = data.unique(subset=["_id"])

# keep top 50 codes
data = keep_top_k_codes(data, ["diagnosis_codes", "procedure_codes"], 50)

# filter out rows with no codes
data = data.filter(
    pl.col("diagnosis_codes").is_not_null() | pl.col("procedure_codes").is_not_null()
)

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

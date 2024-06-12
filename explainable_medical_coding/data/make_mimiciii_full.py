"""
Script to create the mimiciii_full dataset from the mimiciii dataset. The dataset is from the Mullenbach et al. paper.
"""

from pathlib import Path

import polars as pl

new_folder_path = Path("data/processed/mimiciii_full")
# create folder
new_folder_path.mkdir(parents=True, exist_ok=True)

data = pl.read_parquet("data/processed/mimiciii.parquet")
splits = pl.read_ipc("data/splits/mimiciii_full_splits.feather")
data = data.join(splits, on="_id")

# remove not used columns
data = data.drop(["note_seq", "charttime", "storetime"])


data = data.filter(
    (pl.col("diagnosis_code_type") == "icd9cm")
    | (pl.col("procedure_code_type") == "icd9pcs")
)

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

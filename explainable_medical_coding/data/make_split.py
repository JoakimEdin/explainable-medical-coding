# -*- coding: utf-8 -*-
import logging
import random
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split

from explainable_medical_coding.utils.settings import (
    ID_COLUMN,
    SUBJECT_ID_COLUMN,
    TARGET_COLUMN,
)

TEST_SIZE = 0.15
VAL_SIZE = 0.1
random.seed(10)


@click.command()
@click.argument("output_filepath_str", type=click.Path())
def main(output_filepath_str: str):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info("making split using stratified sampling")
    output_filepath = Path(output_filepath_str)
    # save files to disk
    mimiciv_10 = pd.read_feather(output_filepath / "mimiciv_icd10.feather")

    mimiciv_10[TARGET_COLUMN] = mimiciv_10[TARGET_COLUMN].apply(lambda x: list(x))
    splits = mimiciv_10[[SUBJECT_ID_COLUMN, ID_COLUMN]]

    subject_series = mimiciv_10.groupby(SUBJECT_ID_COLUMN)[TARGET_COLUMN].sum()
    X = subject_series.index.values.reshape(-1, 1)
    targets = subject_series.values
    enc = MultiLabelBinarizer(sparse_output=False)
    y = enc.fit_transform(targets)

    X_train_val, y_train_val, X_test, y_test = iterative_train_test_split(
        X, y, test_size=TEST_SIZE
    )
    logging.info("Created test set")
    X_train, y_train, X_val, y_val = iterative_train_test_split(
        X_train_val, y_train_val, test_size=VAL_SIZE / (1 - TEST_SIZE)
    )

    X_train = X_train.reshape(-1)
    X_val = X_val.reshape(-1)
    X_test = X_test.reshape(-1)

    # Save the data
    splits.loc[splits[SUBJECT_ID_COLUMN].isin(X_train), "split"] = "train"
    splits.loc[splits[SUBJECT_ID_COLUMN].isin(X_val), "split"] = "val"
    splits.loc[splits[SUBJECT_ID_COLUMN].isin(X_test), "split"] = "test"

    splits = splits[[ID_COLUMN, "split"]].reset_index(drop=True)

    logging.info("Test Size: %f", len(X_test) / len(X))
    logging.info("Val Size: %f", len(X_val) / len(X))
    logging.info("Train Size: %f", len(X_train) / len(X))

    logging.info("Labels missing in the test set: %f", (y_test.sum(axis=0) == 0).mean())
    logging.info("Labels missing in the val set: %f", (y_val.sum(axis=0) == 0).mean())
    logging.info(
        "Labels missing in the train set: %f", (y_train.sum(axis=0) == 0).mean()
    )

    splits.to_feather(output_filepath / "splits.feather")
    logging.info("Saved splits to disk")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

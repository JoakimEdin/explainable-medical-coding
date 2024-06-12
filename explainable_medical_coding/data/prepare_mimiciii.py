"""
This script takes the raw data from the MIMIC-III dataset and prepares it for automatic medical coding.
The script does the following:
1. Loads the data from the csv files.
2. Renames the columns to match the column names in the MIMIC-IV dataset.
3. Adds punctuations to the ICD-9-CM and ICD-9-PCS codes.
4. Joins discharge summaries with addendums.
5. Removes duplicate rows.
6. Removes cases with no codes.
7. Saves the data as parquet files.

MIMIC-III have many different types of notes that are ignored in this script.
The notes are stored in the NOTEEVENTS.csv file. Here are the note categories and their counts:
┌───────────────────┬────────┐
│ CATEGORY          ┆ counts │
│ ---               ┆ ---    │
│ str               ┆ u32    │
╞═══════════════════╪════════╡
│ Discharge summary ┆ 59652  │
│ Physician         ┆ 141624 │
│ Case Management   ┆ 967    │
│ Consult           ┆ 98     │
│ Nursing           ┆ 223556 │
│ General           ┆ 8301   │
│ Respiratory       ┆ 31739  │
│ Echo              ┆ 45794  │
│ Social Work       ┆ 2670   │
│ Radiology         ┆ 522279 │
│ Nursing/other     ┆ 822497 │
│ Nutrition         ┆ 9418   │
│ Pharmacy          ┆ 103    │
│ ECG               ┆ 209051 │
│ Rehab Services    ┆ 5431   │
└───────────────────┴────────┘
These notes may be useful for other tasks. For example, for pre-training language models.
It is also not guaranteed that all the information is in the discharge summaries.
"""
import logging
import random
from pathlib import Path

import click
import polars as pl
from dotenv import find_dotenv, load_dotenv

from explainable_medical_coding.utils.data_helper_functions import (
    reformat_icd9cm_code,
    reformat_icd9pcs_code,
)
from explainable_medical_coding.utils.settings import (
    ID_COLUMN,
    SUBJECT_ID_COLUMN,
    TEXT_COLUMN,
)

random.seed(10)


def parse_code_dataframe(
    df: pl.DataFrame,
    code_column: str = "diagnosis_codes",
    code_type_column: str = "diagnosis_code_type",
) -> pl.DataFrame:
    """Change names of colums, remove duplicates and Nans, and takes a dataframe and a column name
    and returns a series with the column name and a list of codes.

    Example:
        Input:
                subject_id  _id     target
                       2   163353     V3001
                       2   163353      V053
                       2   163353      V290

        Output:
            target    [V053, V290, V3001]

    Args:
        row (pd.DataFrame): Dataframe with a column of codes.
        col (str): column name of the codes.

    Returns:
        pd.Series: Series with the column name and a list of codes.
    """

    df = df.filter(df[code_column].is_not_null())
    df = df.unique(subset=[ID_COLUMN, code_column])
    df = df.group_by([ID_COLUMN, code_type_column]).agg(
        pl.col(code_column).map_elements(list).alias(code_column)
    )
    return df


def parse_notes_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Parse the notes dataframe by filtering out notes with no text and removing duplicates."""
    df = df.filter(df["note_type"] == "Discharge summary")
    df = df.filter(df[TEXT_COLUMN].is_not_null())
    df = df.sort(
        [ID_COLUMN, "note_subtype", "CHARTTIME", "CHARTDATE", "note_id"],
        descending=[False, True, False, False, False],
    )
    # join the notes with the same id and note type. This is to join discharge summaries and addendums.
    df = df.group_by(SUBJECT_ID_COLUMN, ID_COLUMN, "note_type").agg(
        pl.col("text").str.concat(" "),
        pl.col("note_subtype").str.to_lowercase().str.concat("+"),
        pl.col("note_id").str.concat("+"),
    )
    return df


@click.command()
@click.argument("input_filepath_str", type=click.Path(exists=True))
@click.argument("output_filepath_str", type=click.Path())
def main(input_filepath_str: str, output_filepath_str: str):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    input_filepath = Path(input_filepath_str)
    output_filepath = Path(output_filepath_str)
    output_filepath.mkdir(parents=True, exist_ok=True)

    # Load the dataframes
    mimic_notes = pl.read_csv(
        input_filepath / "physionet.org/files/mimiciii/1.4/NOTEEVENTS.csv.gz"
    )
    mimic_diag = pl.read_csv(
        input_filepath / "physionet.org/files/mimiciii/1.4/DIAGNOSES_ICD.csv.gz",
        dtypes={"ICD9_CODE": str},
    )
    mimic_proc = pl.read_csv(
        input_filepath / "physionet.org/files/mimiciii/1.4/PROCEDURES_ICD.csv.gz",
        dtypes={"ICD9_CODE": str},
    )

    # rename the columns
    mimic_notes = mimic_notes.rename(
        {
            "HADM_ID": ID_COLUMN,
            "SUBJECT_ID": SUBJECT_ID_COLUMN,
            "ROW_ID": "note_id",
            "TEXT": TEXT_COLUMN,
            "CATEGORY": "note_type",
            "DESCRIPTION": "note_subtype",
        }
    )
    mimic_diag = mimic_diag.rename(
        {
            "HADM_ID": ID_COLUMN,
            "ICD9_CODE": "diagnosis_codes",
        }
    ).drop(["SUBJECT_ID"])
    mimic_proc = mimic_proc.rename(
        {
            "HADM_ID": ID_COLUMN,
            "ICD9_CODE": "procedure_codes",
        }
    ).drop(["SUBJECT_ID"])

    # Format the code type columns
    mimic_diag = mimic_diag.with_columns(diagnosis_code_type=pl.lit("icd9cm"))

    mimic_proc = mimic_proc.with_columns(procedure_code_type=pl.lit("icd9pcs"))

    # Format the diagnosis codes by adding punctuations
    mimic_diag = mimic_diag.with_columns(
        pl.col("diagnosis_codes").map_elements(reformat_icd9cm_code)
    )
    mimic_proc = mimic_proc.with_columns(
        pl.col("procedure_codes").map_elements(reformat_icd9pcs_code)
    )

    # Process codes and notes
    mimic_diag = parse_code_dataframe(
        mimic_diag,
        code_column="diagnosis_codes",
        code_type_column="diagnosis_code_type",
    )

    mimic_proc = parse_code_dataframe(
        mimic_proc,
        code_column="procedure_codes",
        code_type_column="procedure_code_type",
    )
    mimic_notes = parse_notes_dataframe(mimic_notes)
    mimic_codes = mimic_diag.join(mimic_proc, on=ID_COLUMN, how="outer_coalesce")
    mimiciii = mimic_notes.join(mimic_codes, on=ID_COLUMN, how="inner")

    # save files to disk
    mimiciii.write_parquet(output_filepath / "mimiciii.parquet")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

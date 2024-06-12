# Lint as: python3
"""MIMIC-III-clean: A public medical coding dataset from MIMIC-III with ICD-9 diagnosis and procedure codes."""

import datasets
import polars as pl

from explainable_medical_coding.utils.settings import REPOSITORY_PATH

logger = datasets.logging.get_logger(__name__)


_CITATION = """
@inproceedings{edinAutomatedMedicalCoding2023,
  address = {Taipei, Taiwan},
  title = {Automated {Medical} {Coding} on {MIMIC}-{III} and {MIMIC}-{IV}: {A} {Critical} {Review} and {Replicability} {Study}},
  isbn = {978-1-4503-9408-6},
  shorttitle = {Automated {Medical} {Coding} on {MIMIC}-{III} and {MIMIC}-{IV}},
  doi = {10.1145/3539618.3591918},
  booktitle = {Proceedings of the 46th {International} {ACM} {SIGIR} {Conference} on {Research} and {Development} in {Information} {Retrieval}},
  publisher = {ACM Press},
  author = {Edin, Joakim and Junge, Alexander and Havtorn, Jakob D. and Borgholt, Lasse and Maistro, Maria and Ruotsalo, Tuukka and Maaløe, Lars},
  year = {2023}
}
"""

_DESCRIPTION = """
MIMIC-III-clean: A medical coding dataset from the paper: Automated Medical Coding on MIMIC-III and MIMIC-IV: A Critical Review and Replicability Study.
The dataset is created from MIMIC-IV containing ICD-9-CM and ICD-9-PCS codes. You can obtain the license in https://physionet.org/content/mimiciii/1.4/.
"""

_URL = REPOSITORY_PATH / "data" / "processed" / "mimiciii_clean"
_URLS = {
    "train": _URL / "train.parquet",
    "val": _URL / "val.parquet",
    "test": _URL / "test.parquet",
}


class MIMIC_III_Clean_Config(datasets.BuilderConfig):
    """BuilderConfig for MIMIC-III-clean."""

    def __init__(self, **kwargs):
        """BuilderConfig for MIMIC-III-clean.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MIMIC_III_Clean_Config, self).__init__(**kwargs)


class MIMIC_III_Clean(datasets.GeneratorBasedBuilder):
    """MIMIC-III-Clean: A public medical coding dataset from MIMIC-III with ICD-9 diagnosis and procedure codes Version 1.0"""

    BUILDER_CONFIGS = [
        MIMIC_III_Clean_Config(
            name="mimiciii-clean",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "subject_id": datasets.Value("int64"),
                    "_id": datasets.Value("int64"),
                    "note_type": datasets.Value("string"),
                    "note_id": datasets.Value("string"),
                    "note_subtype": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "diagnosis_codes": datasets.Sequence(datasets.Value("string")),
                    "diagnosis_code_type": datasets.Value("string"),
                    "procedure_codes": datasets.Sequence(datasets.Value("string")),
                    "procedure_code_type": datasets.Value("string"),
                }
            ),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["val"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": downloaded_files["test"]},
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        key = 0
        dataframe = pl.read_parquet(filepath)

        for row in dataframe.to_dicts():
            yield row["_id"], row
            key += 1

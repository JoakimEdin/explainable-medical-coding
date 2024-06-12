# Lint as: python3
"""MIMIC-IV-ICD9: A public medical coding dataset from MIMIC-IV with ICD-9 diagnosis and procedure codes."""

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
MIMIC-IV-ICD9: A medical coding dataset from Automated Medical Coding on MIMIC-III and MIMIC-IV: A Critical Review and Replicability Study.
The dataset is created from MIMIC-IV containing ICD-9-CM and ICD-9-PCS codes. You can obtain the license in https://physionet.org/content/mimiciv/2.2/.
"""

_URL = REPOSITORY_PATH / "data" / "processed" / "mimiciv_icd9"
_URLS = {
    "train": _URL / "train.parquet",
    "val": _URL / "val.parquet",
    "test": _URL / "test.parquet",
}


class MIMIC_IV_ICD9_Config(datasets.BuilderConfig):
    """BuilderConfig for MIMIC-IV-ICD9."""

    def __init__(self, **kwargs):
        """BuilderConfig for MIMIC-IV-ICD9.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MIMIC_IV_ICD9_Config, self).__init__(**kwargs)


class MIMIC_IV_ICD9(datasets.GeneratorBasedBuilder):
    """MIMIC-IV-ICD9: A public medical coding dataset from MIMIC-IV with ICD-9 diagnosis and procedure codes Version 1.0"""

    BUILDER_CONFIGS = [
        MIMIC_IV_ICD9_Config(
            name="mimic_iv_icd9",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "note_id": datasets.Value("string"),
                    "subject_id": datasets.Value("int64"),
                    "_id": datasets.Value("int64"),
                    "note_type": datasets.Value("string"),
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

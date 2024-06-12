# Lint as: python3
"""MIMIC-III-50: A public medical coding dataset from MIMIC-III with ICD-9 diagnosis and procedure codes."""

import datasets
import polars as pl

from explainable_medical_coding.utils.settings import REPOSITORY_PATH

logger = datasets.logging.get_logger(__name__)


_CITATION = """
@inproceedings{mullenbach-etal-2018-explainable,
    title = "Explainable Prediction of Medical Codes from Clinical Text",
    author = "Mullenbach, James  and
      Wiegreffe, Sarah  and
      Duke, Jon  and
      Sun, Jimeng  and
      Eisenstein, Jacob",
    booktitle = "Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)",
    month = jun,
    year = "2018",
    address = "New Orleans, Louisiana",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N18-1100",
    doi = "10.18653/v1/N18-1100",
    pages = "1101--1111",
    abstract = "Clinical notes are text documents that are created by clinicians for each patient encounter. They are typically accompanied by medical codes, which describe the diagnosis and treatment. Annotating these codes is labor intensive and error prone; furthermore, the connection between the codes and the text is not annotated, obscuring the reasons and details behind specific diagnoses and treatments. We present an attentional convolutional network that predicts medical codes from clinical text. Our method aggregates information across the document using a convolutional neural network, and uses an attention mechanism to select the most relevant segments for each of the thousands of possible codes. The method is accurate, achieving precision@8 of 0.71 and a Micro-F1 of 0.54, which are both better than the prior state of the art. Furthermore, through an interpretability evaluation by a physician, we show that the attention mechanism identifies meaningful explanations for each code assignment.",
}
"""

_DESCRIPTION = """
MIMIC-III-50: A medical coding dataset from the Mullenbach et al. (2018) paper. Mullenbach et al. sampled the splits randomly and didn't exclude any rare codes.
We have not processed the text in this dataset. You can obtain the license in https://physionet.org/content/mimiciii/1.4/.
"""

_URL = REPOSITORY_PATH / "data" / "processed" / "mimiciii_50"
_URLS = {
    "train": _URL / "train.parquet",
    "val": _URL / "val.parquet",
    "test": _URL / "test.parquet",
}


class MIMIC_III_50_Config(datasets.BuilderConfig):
    """BuilderConfig for MIMIC-III-50."""

    def __init__(self, **kwargs):
        """BuilderConfig for MIMIC-III-50.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MIMIC_III_50_Config, self).__init__(**kwargs)


class MIMIC_III_50(datasets.GeneratorBasedBuilder):
    """MIMIC-III-50: A public medical coding dataset from MIMIC-III with ICD-9 diagnosis and procedure codes Version 1.0"""

    BUILDER_CONFIGS = [
        MIMIC_III_50_Config(
            name="mimiciii-50",
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

# Lint as: python3
"""MDACE-Profee-ICD10"""


import datasets
import polars as pl

from explainable_medical_coding.utils.settings import REPOSITORY_PATH

logger = datasets.logging.get_logger(__name__)


_CITATION = """
@inproceedings{cheng-etal-2023-mdace,
    title = "{MDACE}: {MIMIC} Documents Annotated with Code Evidence",
    author = "Cheng, Hua  and
      Jafari, Rana  and
      Russell, April  and
      Klopfer, Russell  and
      Lu, Edmond  and
      Striner, Benjamin  and
      Gormley, Matthew",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.416",
    pages = "7534--7550",
    abstract = "We introduce a dataset for evidence/rationale extraction on an extreme multi-label classification task over long medical documents. One such task is Computer-Assisted Coding (CAC) which has improved significantly in recent years, thanks to advances in machine learning technologies. Yet simply predicting a set of final codes for a patient encounter is insufficient as CAC systems are required to provide supporting textual evidence to justify the billing codes. A model able to produce accurate and reliable supporting evidence for each code would be a tremendous benefit. However, a human annotated code evidence corpus is extremely difficult to create because it requires specialized knowledge. In this paper, we introduce MDACE, the first publicly available code evidence dataset, which is built on a subset of the MIMIC-III clinical records. The dataset {--} annotated by professional medical coders {--} consists of 302 Profee charts with 3,1034 evidence spans and 52 Profee charts with 5,563 evidence spans. We implemented several evidence extraction methods based on the EffectiveCAN model (Liu et al., 2021) to establish baseline performance on this dataset. MDACE can be used to evaluate code evidence extraction methods for CAC systems, as well as the accuracy and interpretability of deep learning models for multi-label classification. We believe that the release of MDACE will greatly improve the understanding and application of deep learning technologies for medical coding and document classification.",
}
"""

_DESCRIPTION = """
MDACE Profee ICD-10: MDACE is subset of MIMIC-III. Medical coders re-annotated the MIMIC-III dataset with code evidence spans. MDACE is from the Cheng et al. paper. MDACE Profee ICD-10 is a subset of MDACE comprising only the Profee charts with ICD-10 codes.
We have not processed the text in this dataset. You can obtain the license in https://physionet.org/content/mimiciii/1.4/.
"""

_URL = REPOSITORY_PATH / "data" / "processed" / "mdace_icd10_profee"
_URLS = {
    "train": _URL / "train.parquet",
    "val": _URL / "val.parquet",
    "test": _URL / "test.parquet",
}


class MDACE_Profee_ICD10_Config(datasets.BuilderConfig):
    """BuilderConfig for MDACE-Profee-ICD10."""

    def __init__(self, **kwargs):
        """BuilderConfig for MDACE-Profee-ICD10.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MDACE_Profee_ICD10_Config, self).__init__(**kwargs)


class MDACE_Profee_ICD10(datasets.GeneratorBasedBuilder):
    """MDACE-Profee-ICD10: A public medical coding subset of MIMIC-III cleaned with evidence spans with ICD-10 diagnosis and procedure codes Version 1.0"""

    BUILDER_CONFIGS = [
        MDACE_Profee_ICD10_Config(
            name="mdace_profee_icd10",
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
                    "note_id": datasets.Value("string"),
                    "note_type": datasets.Value("string"),
                    "note_subtype": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "diagnosis_codes": datasets.Sequence(datasets.Value("string")),
                    "diagnosis_code_type": datasets.Value("string"),
                    "diagnosis_code_spans": datasets.Sequence(
                        datasets.Sequence(datasets.Sequence(datasets.Value("int64")))
                    ),
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
            yield row["note_id"], row
            key += 1

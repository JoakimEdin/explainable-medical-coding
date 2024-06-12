# Lint as: python3
"""MDACE-Inpatient-ICD9"""

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
    abstract = "We introduce a dataset for evidence/rationale extraction on an extreme multi-label classification task over long medical documents. One such task is Computer-Assisted Coding (CAC) which has improved significantly in recent years, thanks to advances in machine learning technologies. Yet simply predicting a set of final codes for a patient encounter is insufficient as CAC systems are required to provide supporting textual evidence to justify the billing codes. A model able to produce accurate and reliable supporting evidence for each code would be a tremendous benefit. However, a human annotated code evidence corpus is extremely difficult to create because it requires specialized knowledge. In this paper, we introduce MDACE, the first publicly available code evidence dataset, which is built on a subset of the MIMIC-III clinical records. The dataset {--} annotated by professional medical coders {--} consists of 302 Inpatient charts with 3,934 evidence spans and 52 Profee charts with 5,563 evidence spans. We implemented several evidence extraction methods based on the EffectiveCAN model (Liu et al., 2021) to establish baseline performance on this dataset. MDACE can be used to evaluate code evidence extraction methods for CAC systems, as well as the accuracy and interpretability of deep learning models for multi-label classification. We believe that the release of MDACE will greatly improve the understanding and application of deep learning technologies for medical coding and document classification.",
}
"""

_DESCRIPTION = """
MDACE Inpatient ICD-9: MDACE is subset of MIMIC-III. Medical coders re-annotated the MIMIC-III dataset with code evidence spans. MDACE is from the Cheng et al. paper. MDACE Inpatient ICD-9 is a subset of MDACE comprising only the inpatient charts with ICD-9 codes.
We have not processed the text in this dataset. You can obtain the license in https://physionet.org/content/mimiciii/1.4/.
"""

_URL = REPOSITORY_PATH / "data" / "processed" / "mdace_icd9_inpatient"
_URLS = {
    "train": _URL / "train.parquet",
    "val": _URL / "val.parquet",
    "test": _URL / "test.parquet",
}


class MDACE_Inpatient_ICD9_Config(datasets.BuilderConfig):
    """BuilderConfig for MDACE-Inpatient-ICD9."""

    def __init__(self, **kwargs):
        """BuilderConfig for MDACE-Inpatient-ICD9.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MDACE_Inpatient_ICD9_Config, self).__init__(**kwargs)


class MDACE_Inpatient_ICD9(datasets.GeneratorBasedBuilder):
    """MDACE-Inpatient-ICD9: A public medical coding subset of MIMIC-III cleaned with evidence spans with ICD-9 diagnosis and procedure codes Version 1.0"""

    BUILDER_CONFIGS = [
        MDACE_Inpatient_ICD9_Config(
            name="mdace_inpatient_icd9",
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
                    "procedure_codes": datasets.Sequence(datasets.Value("string")),
                    "procedure_code_type": datasets.Value("string"),
                    "procedure_code_spans": datasets.Sequence(
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

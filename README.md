Explainable Medical Coding
==============================

# Setup
While our paper only presented results on MIMIC-III and MDACE, our code also supports experiments on MIMIC-IV. Here is a guide to setting up the repository for experimentation and reproducibility. Notice that you will need +100 GB of storage to fit everything.
1. Clone this repository.
2. cd explainable_medical_coding
3. `cd data/raw`
4. Install [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/) using wget (we used version 2.2)
5. Install [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/) using wget (we used version 2.2)
6. Install [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) using wget (we used version 1.4)
7. Back to the main repository folder `cd -`
8. Use a virtual environment (e.g., conda) with python 3.11.5 installed.
9. Create a weights and biases account. It is possible to run the experiments without wandb.
10. Prepare code, datasets and models using the command: `make prepare_everything`. Go grab an enourmous coffee.

You are now all set to run experiments!

Instead of using `make prepare_everything`, you can run it in multiple steps. This can be useful if you don't have storage for everything. E.g., if you don't need the model weights which takes +70GB of storage.
1. Enter `make setup`. It should install everything you need to use the code.
2. prepare the datasets and download the models using the command
3. Prepare datasets. `make mimiciii`, `make mimiciv`, `make mdace_icd9`.
4. Download RoBERTa-base-PM-M3-Voc which is necessary for training PLM-ICD `make download_roberta`
5. Download the 10 runs of the PGD, IGR, TM, B_S and B_U. They require 70GB of storage. `make download_models` (the command is slow to execute)


# Note on licenses

## MDAce
We have copied the annotations from https://github.com/3mcloud/MDACE. It is not an attempt to steal credit from the authors, we just want to make the setup of the code as effortless as possible. If you use the annotations, remember to cite the authors:

```
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
```
## MIMIC
You need to obtain a non-commercial licence from physionet to use MIMIC. You will need to complete training. The training is free, but takes a couple of hours. - [link to data access](https://physionet.org/content/mimiciii/1.4/)

## Model weights can only be used non-commercially
While we would love to make everything fully open source, we cannot. Becaue MIMIC has a non-commercial license, the models trained using that data will also have a non-commercial licence. Therefore, using our models or RoBERTa-base-PM-M3-Voc's weights for commercial usecases is forbidden.

# How to run experiments
## How to train a model
You can run any experiment found in `explainable_medical_coding/configs/experiment`. Here are some examples:
   * Train PLM-ICD on MIMIC-III full and MDACE on GPU 0: `poetry run python train_plm.py experiment=mdace_icd9_code/plm_icd gpu=0`
   * Train PLM-ICD using the supervised approach proposed by Cheng et al. on MIMIC-III full and MDACE on GPU 0: `poetry run python train_plm.py experiment=mdace_icd9_code/plm_icd_supervised gpu=0`
   * Train PLM-ICD using input gradient regularization on MIMIC-III full and MDACE on GPU 0: `poetry run python train_plm.py experiment=mdace_icd9_code/plm_icd_igr gpu=0`
   * Train PLM-ICD using token masking on MIMIC-III full and MDACE on GPU 0: `poetry run python train_plm.py experiment=mdace_icd9_code/plm_icd_tm gpu=0`
   * Train PLM-ICD using projected gradient descent on MIMIC-III full and MDACE on GPU 0: `poetry run python train_plm.py experiment=mdace_icd9_code/plm_icd_pgd gpu=0`
   * Train PLM-ICD on MIMIC-III full and MDACE on GPU 0 using a batch_size of 1: `poetry run python train_plm.py experiment=mdace_icd9_code/plm_icd gpu=0 dataloader.max_batch_size=1`
   * Train PLM-ICD on MIMIC-IV ICD-10 and MDACE on GPU 0: `poetry run python train_plm.py experiment=mdace_icd9_code/plm_icd gpu=0 dataloader.max_batch_size=1 data=mimiciv_icd10`


# Evaluation of feature attribution methods
We also use hydra condig file for evaluating the feature attribution methods methods. The config file is found in `explainable_medical_coding/configs/explainability.yaml`. In the config file, you can chose which explanation methods you would like to use and using which model. The script expects the model weights to be in the models folder. Here are some examples:
* Evaluate using all the feature attribution methods in the config file on the model weights found in the models/unsupervised/gice8s68. Store the results in a folder results/explainability_results/baseline: `poetry run python eval_explanations.py gpu=0 run_id=unsupervised/gice8s68 model_name=baseline`
* Only evaluate AttInGrad and Attention: `poetry run python eval_explanations.py gpu=0 run_id=unsupervised/gice8s68 model_name=baseline explainers=[grad_attention, laat]`
* Only evaluate AttInGrad and don't evaluate faithfulness (which is a slow evaluation metric): `poetry run python eval_explanations.py gpu=0 run_id=unsupervised/gice8s68 model_name=baseline explainers=[grad_attention] evaluate_faithfulness=False`
* evaluate multiple models sequentially: `poetry run python eval_explanations.py gpu=0 --multirun run_id=unsupervised/jdjr2y77,unsupervised/pati4i3b,unsupervised/ov55kelz,unsupervised/l2qznkbe model_name=baseline`


# Overview of the repository
#### configs
We use [Hydra](https://hydra.cc/docs/intro/) for configurations. The configs for every experiment is found in `explainable_medical_coding/configs/experiments`. Furthermore, the configuration for the sweeps are found in `explainable_medical_coding/configs/sweeps`. We used [Weights and Biases Sweeps](https://docs.wandb.ai/guides/sweeps) for most of our experiments.

#### data
This is where the splits and datasets are stored

#### models
The directory contains the model weights.

#### reports
This is the code used to generate the plots and tables used in the paper. The code uses the Weights and Biases API to fetch the experiment results. The code is not usable by others, but was included for the possibility to validate our figures and tables.

#### explainable_medical_coding
This is were the code for running the experiments and evaluating explanation methods are found.

# My setup
I ran the experiments on one A100 80GB per experiment. I had 2TB RAM on my machine.

# ⚠️ Known issues
* IGR, TM and PGD require a lot of gpu memory and compute to train. Smaller machines may not be capable of training them. You can use a smaller batch-size using the --max_batch_size parameter. However, a small machine may not fit a batch size of 1 for these adversarial training strategies.

# Acknowledgement
Thank you _ for providing the template for making the datasets in explainable_medical_coding/datasets/.

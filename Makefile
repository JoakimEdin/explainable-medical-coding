.PHONY: clean data

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = explainable_medical_coding
PYTHON_INTERPRETER = python

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################


setup:
	pip install poetry
	poetry config virtualenvs.in-project true
	poetry install
	poetry run pre-commit install

## Make Dataset
mimiciv:
	poetry run $(PYTHON_INTERPRETER) explainable_medical_coding/data/prepare_mimiciv.py data/raw data/processed
	poetry run $(PYTHON_INTERPRETER) explainable_medical_coding/data/make_mimiciv_icd10.py
	poetry run $(PYTHON_INTERPRETER) explainable_medical_coding/data/make_mimiciv_icd9.py

mimiciii:
	poetry run $(PYTHON_INTERPRETER) explainable_medical_coding/data/prepare_mimiciii.py data/raw data/processed
	poetry run $(PYTHON_INTERPRETER) explainable_medical_coding/data/make_mimiciii_clean.py
	poetry run $(PYTHON_INTERPRETER) explainable_medical_coding/data/make_mimiciii_full.py
	poetry run $(PYTHON_INTERPRETER) explainable_medical_coding/data/make_mimiciii_50.py

mdace:
	poetry run $(PYTHON_INTERPRETER) explainable_medical_coding/data/prepare_mdace.py data/raw data/processed
	poetry run $(PYTHON_INTERPRETER) explainable_medical_coding/data/make_mdace_icd9_inpatient.py
	poetry run $(PYTHON_INTERPRETER) explainable_medical_coding/data/make_mdace_icd9_inpatient_code.py
	poetry run $(PYTHON_INTERPRETER) explainable_medical_coding/data/make_mdace_icd10_inpatient.py

download_roberta:
	wget https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-M3-Voc-hf.tar.gz -P models
	tar -xvzf models/RoBERTa-base-PM-M3-Voc-hf.tar.gz -C models
	rm models/RoBERTa-base-PM-M3-Voc-hf.tar.gz
	mv models/RoBERTa-base-PM-M3-Voc/RoBERTa-base-PM-M3-Voc-hf models/roberta-base-pm-m3-voc-hf
	rm -r models/RoBERTa-base-PM-M3-Voc

download_models:
	poetry run gdown --id 1Gna1tEQqtSrBQC_2QX_1HGpZBYAFQCix -O models/temp.tar.gz
	tar -xvzf models/temp.tar.gz
	rm models/temp.tar.gz

prepare_everything:
	make setup
	make mimiciv
	make mimiciii
	make mdace
	make download_roberta
	make download_models

import pandas as pd
import torch
from datasets import Dataset
from omegaconf import OmegaConf

from explainable_medical_coding.utils.tokenizer import TargetTokenizer
from explainable_medical_coding.utils.datatypes import Batch
from explainable_medical_coding.utils.settings import TEXT_COLUMN, TARGET_COLUMN


def preprocess_dataframe(config: OmegaConf, dataframe: pd.DataFrame) -> pd.DataFrame:
    text_split = dataframe[TEXT_COLUMN].str.split()
    dataframe["num_words"] = text_split.apply(len)
    # # truncate text to max length
    # dataframe[TEXT_COLUMN] = text_split.str.slice(0, config.max_length).str.join(" ")
    # replace three underscores with mask token
    if config.replace_underscore_with_mask:
        dataframe[TEXT_COLUMN] = dataframe[TEXT_COLUMN].str.replace("___", "<mask>")
    # else:
    #     dataframe[TEXT_COLUMN] = dataframe[TEXT_COLUMN].str.replace("___", "")

    if config.lowercase:
        dataframe[TEXT_COLUMN] = dataframe[TEXT_COLUMN].str.lower()

    if config.remove_punctuation:
        dataframe[TEXT_COLUMN] = dataframe[TEXT_COLUMN].str.replace(
            "[^A-Za-z0-9]+", " ", regex=True
        )

    return dataframe


def one_hot_encode_evidence(
    evidence_input_ids: list[torch.Tensor], num_tokens: int
) -> torch.Tensor:
    batch_size = len(evidence_input_ids)
    one_hot_output = torch.zeros((batch_size, num_tokens))
    for idx in range(batch_size):
        one_hot_output[idx, evidence_input_ids[idx]] = 1
    return one_hot_output


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        target_tokenizer: TargetTokenizer,
        pad_token_id: int = 1,
    ) -> None:
        super().__init__()
        self.hf_dataset = hf_dataset.sort("length")
        self.pad_token_id = pad_token_id
        self.target_tokenizer = target_tokenizer

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> int:
        return idx

    @staticmethod
    def pad(sequence: list[list[int]], pad_id: int) -> torch.Tensor:
        return torch.nn.utils.rnn.pad_sequence(
            sequence, batch_first=True, padding_value=pad_id
        )

    def collate_fn(self, indices: list[int]) -> Batch:
        batch = self.hf_dataset.select(indices)

        input_ids = self.pad(sequence=batch["input_ids"], pad_id=self.pad_token_id)
        target_ids_list = batch["target_ids"]
        if self.target_tokenizer.autoregressive:
            # Check if sos, eos or pad tokens are None types
            if (
                self.target_tokenizer.sos_id is None
                or self.target_tokenizer.eos_id is None
                or self.target_tokenizer.pad_id is None
            ):
                raise ValueError(
                    "sos_id, eos_id and pad_id must be defined for autoregressive models"
                )
            # add sos and eos tokens

            target_ids_list = [
                torch.cat(
                    [
                        torch.tensor([self.target_tokenizer.sos_id]),
                        target_ids,
                        torch.tensor([self.target_tokenizer.eos_id]),
                    ]
                )
                for target_ids in target_ids_list
            ]
            targets = self.pad(
                sequence=target_ids_list, pad_id=self.target_tokenizer.pad_id
            )
        else:
            targets = self.target_tokenizer.torch_one_hot_encoder(target_ids_list)

        attention_masks = self.pad(sequence=batch["attention_mask"], pad_id=0)
        lengths = batch["length"]
        ids = batch["_id"]
        texts = batch[TEXT_COLUMN]
        target_names = batch[TARGET_COLUMN]
        if "evidence_input_ids" in batch.column_names:
            evidence_input_ids = batch["evidence_input_ids"]
        else:
            evidence_input_ids = None

        if "note_id" in batch.column_names:
            note_ids = batch["note_id"]
        else:
            note_ids = None

        if "teacher_logits" in batch.column_names:
            teacher_logits = batch["teacher_logits"]
        else:
            teacher_logits = None

        return Batch(
            input_ids=input_ids,
            targets=targets,
            target_names=target_names,
            attention_masks=attention_masks,
            lengths=lengths,
            ids=ids,
            note_ids=note_ids,
            texts=texts,
            evidence_input_ids=evidence_input_ids,
            teacher_logits=teacher_logits,
        )

import json
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase, RobertaTokenizer

from explainable_medical_coding.utils.settings import EOS, PAD, SOS


class TargetTokenizer:
    """TargetTokenizer class for tokenizing target sequences."""

    def __init__(
        self,
        autoregressive: bool = False,
        output_torch: bool = True,
        one_hot: bool = True,
    ) -> None:
        """Initialize the TargetTokenizer.

        Args:
            autoregressive (bool, optional): Whether to add SOS, EOS and PAD tokens. Defaults to False.
            output_torch (bool, optional): Whether to output torch tensors. Defaults to True.
        """
        self.target2id: dict[str, int] = {}
        self.id2target: list[str] = []
        self.autoregressive = autoregressive
        self.output_torch = output_torch

    def fit(self, targets: list[str]) -> None:
        """Fit the tokenizer to the targets.

        Args:
            targets (list[str]): The targets.
        """
        self.id2target = sorted(list(set(targets)))
        if self.autoregressive:
            self.id2target = [SOS, EOS, PAD] + self.id2target
        self.target2id = {target: idx for idx, target in enumerate(self.id2target)}

    def filter_unknown_targets(self, targets: list[str]) -> list[str]:
        """Filter the unknown targets.

        Args:
            targets (list[str]): The targets.

        Returns:
            list[int]: The filtered targets.
        """
        return [target for target in targets if target in self.target2id]

    def encode(self, targets: list[str]) -> list[int] | torch.Tensor:
        """Encode the targets.

        Args:
            targets (list[str]): The targets.

        Returns:
            list[int]: The encoded targets.
        """

        encoded_targets = [self.target2id[target] for target in targets]
        if self.output_torch:
            return torch.tensor(encoded_targets, dtype=torch.int64)
        return encoded_targets

    def decode(self, ids: list[int]) -> list[str]:
        """Decode the ids."""
        return [self.id2target[idx] for idx in ids]

    def __len__(self) -> int:
        return len(self.id2target)

    def __getitem__(self, idx: int) -> str:
        return self.id2target[idx]

    def __contains__(self, target: str) -> bool:
        return target in self.target2id

    def __iter__(self) -> Iterable[str]:
        return iter(self.id2target)

    def __repr__(self) -> str:
        return f"TargetTokenizer(autoregressive={self.autoregressive})"

    def __call__(self, targets: list[str]) -> list[int] | torch.Tensor:
        return self.encode(targets)

    @property
    def eos_id(self) -> Optional[int]:
        return self.target2id.get(EOS)

    @property
    def sos_id(self) -> Optional[int]:
        return self.target2id.get(SOS)

    @property
    def pad_id(self) -> int:
        return self.target2id.get(PAD, -1)

    def target_names(self) -> list[str]:
        return self.id2target

    def torch_one_hot_encoder(self, target_ids: list[torch.Tensor]) -> torch.Tensor:
        """Encode a list of targets to a one-hot encoded tensor."""
        """Transform a list of target ids into a one-hot encoding.

        Args:
            target_ids (list[torch.Tensor]): list of target ids.
            num_classes (int): number of classes.

        Returns:
            torch.Tensor: one-hot encoding of the target ids.
        """
        target_ids = [
            torch.nn.functional.one_hot(target_id, num_classes=len(self)).sum(0).bool()
            for target_id in target_ids
        ]
        return torch.vstack(target_ids).float()

    def torch_one_hot_decoder(self, tensor: torch.Tensor) -> list[str]:
        """Decode a one-hot encoded tensor."""
        indices = torch.nonzero(tensor).squeeze(0).numpy()
        return list([self.id2target[int(index)] for index in indices])

    def numpy_one_hot_decoder(self, array: np.array) -> list[str]:
        """Decode a one-hot encoded tensor."""

        indices = np.nonzero(array)[1]
        return list([self.id2target[int(index)] for index in indices])

    def save(self, path: Path | str) -> None:
        """Save the tokenizer to a json file."""
        with open(path, "w") as f:
            json.dump(self.id2target, f)

    def load(self, path: Path | str) -> None:
        """Load the tokenizer from a json file."""
        with open(path, "r") as f:
            self.id2target = json.load(f)
        self.target2id = {target: idx for idx, target in enumerate(self.id2target)}


def get_tokens(
    input_ids: torch.Tensor | list, text_tokenizer: PreTrainedTokenizerBase
) -> list[str]:
    return text_tokenizer.convert_ids_to_tokens(
        torch.tensor(input_ids).squeeze().tolist()
    )


def get_word_map_roberta(
    input_ids: torch.Tensor,
    text_tokenizer: RobertaTokenizer,
    include_space: bool = True,
) -> torch.Tensor:
    """Get a list mapping each token to a word. This is necessary because the Roberta tokenizer splits words into subwords.
    The spaces are joined with the preceding word.
    Example:
    input_ids = [0, 1, 2, 3, 4, 5, 6, 7]


    Args:
        input_ids (torch.Tensor): Input ids to get word map for
        text_tokenizer (RobertaTokenizer): Text tokenizer to use

    Returns:
        list[int]: Word map of same length as input_ids
    """
    tokens = get_tokens(input_ids, text_tokenizer)
    word_map = []
    idx = 0
    space_between_words = False
    for token in tokens:
        if re.match(r"^<.*>$", token):
            if idx in word_map:
                idx += 1
            word_map.append(idx)
            idx += 1
            space_between_words = False
            continue

        # if the token only contains Ċ or Ġ characters
        if re.match(r"^[ĊĠ:.,]+$", token):
            space_between_words = True
            if not include_space:
                idx += 1
            word_map.append(idx)
            continue

        if token.startswith("Ġ"):
            space_between_words = True

        if space_between_words:
            idx += 1
            space_between_words = False

        word_map.append(idx)

    return torch.tensor(word_map)


def spans_to_token_ids(
    input_ids: torch.Tensor,
    all_spans: list[list[list[int]]],
    text_tokenizer: PreTrainedTokenizerBase,
) -> list[list[int]]:
    """Takes evidence spans representing character positions and returns token ids

    Args:
        input_ids (torch.Tensor): input ids
        all_spans (list[list[list[int]]]): evidence spans for all codes in an example
        text_tokenizer (PreTrainedTokenizerBase): text tokenizer

    Returns:
        list[list[int]]: token ids
    """
    tokens = get_tokens(input_ids, text_tokenizer)
    token_lengths = torch.tensor(list(map(lambda x: len(x), tokens)))
    token_lengths_no_cls = token_lengths[1:]
    token_lengths_cumsum = token_lengths_no_cls.cumsum(dim=0)
    token_ids = []
    for code_spans in all_spans:
        code_token_ids: list[int] = []
        for span in code_spans:
            start = span[0]
            end = span[1]

            # skip if the span is outside the input_ids
            if start > token_lengths_cumsum[-1]:
                code_token_ids.extend([])
                continue

            start_token_id = (
                torch.where(token_lengths_cumsum - 1 >= start)[0][0].item()
            ) + 1
            end_token_id = torch.where(token_lengths_cumsum >= end)[0][0].item() + 1
            code_token_ids.extend(list(range(start_token_id, end_token_id + 1)))
        token_ids.append(code_token_ids)
    return token_ids


def token_ids_to_spans(
    input_ids: torch.Tensor,
    token_ids: torch.Tensor,
    text_tokenizer: PreTrainedTokenizerBase,
) -> list[list[int]]:
    """Takes token ids and returns spans representing character positions
    Args:
        input_ids (torch.Tensor): input ids
        token_ids (torch.Tensor): token ids
        text_tokenizer (PreTrainedTokenizerBase): text tokenizer

    Returns:
        list[list[int]]: spans
    """
    token_ids = token_ids.sort().values.numpy() - 1  # remove cls token
    tokens = get_tokens(input_ids, text_tokenizer)[1:]  # remove cls token
    token_lengths = torch.tensor(list(map(lambda x: len(x), tokens)))
    token_lengths_cumsum = token_lengths.cumsum(dim=0).numpy()
    start = 0
    end = 0
    new_span = True
    predicted_evidence_spans = []
    for idx in range(len(token_ids)):
        if new_span:
            start_token_id = token_ids[idx] - 1 if token_ids[idx] > 1 else 0
            start = token_lengths_cumsum[start_token_id]
            new_span = False

        if idx == len(token_ids) - 1:
            end = token_lengths_cumsum[token_ids[idx]]
            predicted_evidence_spans.append([start, end])
            break

        if token_ids[idx] + 1 != token_ids[idx + 1]:
            end = token_lengths_cumsum[token_ids[idx]]
            predicted_evidence_spans.append([start, end])
            new_span = True

    return predicted_evidence_spans

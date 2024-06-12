import math
from typing import Optional

import numpy as np
import pandas as pd
import torch

from explainable_medical_coding.utils.data_helper_functions import reformat_icd10cm_code
from explainable_medical_coding.utils.settings import TARGET_COLUMN


def get_code2description() -> dict[str, str]:
    """Get a dictionary mapping ICD codes to descriptions.

    Returns:
        dict[str, str]: Dictionary mapping ICD codes to descriptions
    """
    df_descriptions = pd.read_csv(
        "data/raw/physionet.org/files/mimiciv/2.2/hosp/d_icd_diagnoses.csv.gz",
        compression="gzip",
    )
    df_descriptions = df_descriptions[df_descriptions["icd_version"] == 10]
    df_descriptions = df_descriptions.rename(columns={"icd_code": "target"})
    df_descriptions[TARGET_COLUMN] = df_descriptions[TARGET_COLUMN].apply(
        reformat_icd10cm_code
    )
    return pd.Series(
        df_descriptions["long_title"].values, index=df_descriptions["target"]
    ).to_dict()


@torch.no_grad()
def predict(
    model: torch.nn.Module, input_ids: torch.Tensor, device: str | torch.device
) -> torch.Tensor:
    """Output a model's predictions

    Args:
        model (torch.nn.Module): Model to predict with
        input_ids (torch.Tensor): Input ids to predict on
        device (str | torch.device): Device to run the model on

    Returns:
        np.ndarray: Predictions
    """
    input_ids = input_ids.to(device)
    attention_mask = create_attention_mask(input_ids)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        logits = model(input_ids, attention_mask)
    return torch.sigmoid(logits).detach()


def create_attention_mask(input_ids: torch.Tensor) -> torch.Tensor:
    """Create attention mask for a given input

    Args:
        input_ids (torch.Tensor): Input ids to create attention mask for

    Returns:
        torch.Tensor: Attention mask
    """
    attention_mask = torch.ones_like(input_ids)
    return attention_mask


def create_baseline_input(
    input_ids: torch.Tensor,
    baseline_token_id: int = 50_000,
    cls_token_id: int = 0,
    eos_token_id: int = 2,
) -> torch.Tensor:
    """Create baseline input for a given input

    Args:
        input_ids (torch.Tensor): Input ids to create baseline input for
        baseline_token_id (int, optional): Baseline token id. Defaults to 50_000.
        cls_token_id (int, optional): CLS token id. Defaults to 0.
        eos_token_id (int, optional): SEP token id. Defaults to 2.

    Returns:
        torch.Tensor: Baseline input
    """
    baseline = torch.ones_like(input_ids) * baseline_token_id
    baseline[:, 0] = cls_token_id
    baseline[:, -1] = eos_token_id
    return baseline


def reshape_plm_icd_attributions(
    attributions: torch.Tensor, input_ids: torch.Tensor
) -> torch.Tensor:
    """Reshape attributions to match input_ids. This is necessary because the PLMICD model chunks the input_ids into smaller batches.

    Args:
        attributions (torch.Tensor): Attributions to reshape
        input_ids (torch.Tensor): Input ids to reshape attributions to

    Returns:
        torch.Tensor: Reshaped attributions
    """
    attributions = attributions.reshape(-1, attributions.size(-1))
    padding = attributions.shape[0] - input_ids.shape[-1]
    attributions = attributions[:-padding]
    return attributions.unsqueeze(0)


def embedding_attributions_to_token_attributions(
    attributions: torch.Tensor,
) -> torch.Tensor:
    """Convert embedding attributions to token attributions.

    Args:
        attributions (torch.Tensor): Embedding Attributions,

    Returns:
        torch.Tensor: Token attributions
    """

    return torch.norm(attributions, p=2, dim=-1)


def get_feature_attributions(
    feature_map: torch.Tensor, attributions: torch.Tensor
) -> torch.Tensor:
    """Sum attributions per feature.
    Example:

        feature_map = torch.tensor([0,1,1,2,2])
        attributions = torch.tensor([0.1,0.2,0.3,0.4,0.5])

        Output: torch.tensor([0.1,0.5,0.9])


    Args:
        feature_map (torch.Tensor): Feature map of shape [sequence_length]. Maps each token to a feature.
        attributions (torch.Tensor): Attributions of shape [sequence_length, num_classes]

    Returns:
        torch.Tensor: Feature attributions of shape [num_features, num_classes]
    """
    assert (
        feature_map.shape[0] == attributions.shape[0]
    ), "feature_map must be of same length as sequence_length"
    num_classes = attributions.shape[1]

    # sum attributions per word specified in feature_map
    feature_attributions = torch.zeros((feature_map.max() + 1, num_classes))
    for idx, feature in enumerate(feature_map):
        feature_attributions[feature] += attributions[idx]

    return feature_attributions


class PermuteDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_ids: torch.Tensor,
        attributions: torch.Tensor,
        target_id: Optional[int] = None,
        feature_map: Optional[torch.Tensor] = None,
        baseline_token_id: Optional[int] = None,
        descending: bool = True,
        step_size: int = 1,
        device: str | torch.device = "cpu",
        pad_token_id: int = 1,
        sos_token_id: int = 0,
        eos_token_id: int = 2,
        max_features: Optional[int] = None,
        cumulative: bool = True,
    ):
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id

        pad_token_id_mask = (input_ids != pad_token_id).cpu().squeeze()

        # remove pad tokens, sos and eos tokens. The sos and eos tokens are added before feeding the input to the model.
        self.input_ids = input_ids.cpu()[:, pad_token_id_mask][:, 1:-1]
        self.attributions = attributions.cpu()[pad_token_id_mask][1:-1]

        if feature_map is None:
            self.feature_map = torch.arange(self.input_ids.shape[1])
        else:
            self.feature_map = feature_map.cpu()[pad_token_id_mask][1:-1] - 1

        self.target_id = target_id
        self.baseline_token_id = baseline_token_id
        self.descending = descending
        self.step_size = step_size
        self.device = device
        self.max_features = max_features
        # Get the attributions for each feature instead of for each token.
        self.feature_attributions = get_feature_attributions(
            self.feature_map, self.attributions
        )
        self.cumulative = cumulative

        self.sequence_length = self.input_ids.shape[1]
        if max_features is not None:
            self.num_features = max_features
        else:
            self.num_features = self.feature_attributions.shape[0]

        if target_id is not None:
            target_attributions = self.feature_attributions[:, self.target_id]
            # sort target attributions
            self.sorted_indices = torch.argsort(
                target_attributions, descending=self.descending
            )

    def __len__(self):
        return math.ceil(self.num_features / self.step_size)

    def set_target_id(self, target_id: int):
        self.target_id = target_id
        target_attributions = self.feature_attributions[:, self.target_id]
        # sort target attributions
        self.sorted_indices = torch.argsort(
            target_attributions, descending=self.descending
        )

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.target_id is None:
            raise ValueError("target_id must be set before calling __getitem__")

        permuted_input_ids = self.input_ids.clone()

        if self.cumulative:
            if not self.descending and self.max_features is not None:
                from_idx = (
                    max(len(self.sorted_indices) - self.max_features, 0)
                    + idx * self.step_size
                )
            else:
                from_idx = idx * self.step_size
            feature_indices_to_remove = self.sorted_indices[
                : min(from_idx, len(self.sorted_indices))
            ]
        else:
            feature_indices_to_remove = self.sorted_indices[
                min(idx, len(self.sorted_indices) - 1)
            ].unsqueeze(0)

        # get the indices of the tokens that are mapped to the features in feature_indices_to_remove
        token_indices_to_remove = (
            self.feature_map[:, None] == feature_indices_to_remove
        ).nonzero(as_tuple=True)[0]

        if self.baseline_token_id is None:
            # remove the tokens with the smallest attributions
            token_indices_to_keep = torch.tensor(
                np.delete(
                    np.arange(self.sequence_length),
                    token_indices_to_remove.numpy(),
                    axis=0,
                ),
                device=self.device,
            ).sort()[0]

            # Input IDS but with the tokens not in token_indices_to_keep removed
            permuted_input_ids = self.input_ids[:, token_indices_to_keep]
        else:
            # replace the tokens with the smallest attributions with the baseline token
            permuted_input_ids[:, token_indices_to_remove] = self.baseline_token_id

        # add sos and eos tokens
        permuted_input_ids = torch.cat(
            [
                torch.ones(
                    (permuted_input_ids.shape[0], 1),
                    device=self.device,
                    dtype=permuted_input_ids.dtype,
                )
                * self.sos_token_id,
                permuted_input_ids,
                torch.ones(
                    (permuted_input_ids.shape[0], 1),
                    device=self.device,
                    dtype=permuted_input_ids.dtype,
                )
                * self.eos_token_id,
            ],
            dim=1,
        )
        return permuted_input_ids, idx

    def custom_collate_fn(
        self, batch: list[tuple[torch.Tensor, int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices_list = [b[1] for b in batch]
        input_ids_list = [b[0] for b in batch]
        if self.baseline_token_id is None:
            if len(input_ids_list) > 1:
                raise ValueError("batch_size must be 1 when baseline_token_id is None")
            return input_ids_list[0]

        return torch.vstack(input_ids_list), torch.tensor(indices_list)

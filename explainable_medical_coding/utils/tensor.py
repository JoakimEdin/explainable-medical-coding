import os
from typing import Any

import torch
from omegaconf import OmegaConf
from rich.pretty import pprint


def one_hot(target_ids: list[torch.Tensor], num_classes: int) -> torch.Tensor:
    """Transform a list of target ids into a one-hot encoding.

    Args:
        target_ids (list[torch.Tensor]): list of target ids.
        num_classes (int): number of classes.

    Returns:
        torch.Tensor: one-hot encoding of the target ids.
    """
    target_ids = [
        torch.nn.functional.one_hot(target_id, num_classes=num_classes).sum(0).bool()
        for target_id in target_ids
    ]
    return torch.vstack(target_ids).float()


def detach(tensor: torch.Tensor | Any) -> torch.Tensor | Any:
    """Detach a tensor from the computational graph"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach()
    return tensor


def detach_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Detach a batch from the computational graph"""
    return {k: detach(v) for k, v in batch.items()}


def deterministic() -> None:
    """Run experiment deterministically. There will still be some randomness in the backward pass of the model."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def set_gpu(cfg: OmegaConf) -> Any:
    """Set GPU device."""
    import torch

    # Check if CUDA_VISIBLE_DEVICES is set
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        if cfg.gpu != -1 and cfg.gpu is not None and cfg.gpu != "":
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                ",".join([str(gpu) for gpu in cfg.gpu])
                if isinstance(cfg.gpu, list)
                else str(cfg.gpu)
            )

        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pprint(f"Device: {device}")
    pprint(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    return device

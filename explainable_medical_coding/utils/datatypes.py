from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch

from explainable_medical_coding.utils.tokenizer import TargetTokenizer


@dataclass
class Lookups:
    data_info: dict[str, Any]
    target_tokenizer: TargetTokenizer
    split2code_indices: dict[str, torch.Tensor]


@dataclass
class Batch:
    """Batch class. Used to store a batch of data."""

    input_ids: torch.Tensor
    targets: torch.Tensor
    target_names: list[str]
    ids: list[int]
    note_ids: Optional[list[str]] = None
    lengths: Optional[torch.Tensor] = None
    attention_masks: Optional[torch.Tensor] = None
    texts: Optional[list[str]] = None
    teacher_logits: Optional[torch.Tensor] = None

    evidence_input_ids: Optional[Sequence[Sequence[Sequence[int]]]] = None

    def to(self, device: Any) -> "Batch":
        """Move the batch to a device.

        Args:
            device (Any): Device to move the batch to.

        Returns:
            self: Moved batch.
        """
        self.input_ids = self.input_ids.to(device, non_blocking=True)
        self.targets = self.targets.to(device, non_blocking=True)
        if self.attention_masks is not None:
            self.attention_masks = self.attention_masks.to(device, non_blocking=True)
        if self.teacher_logits is not None:
            self.teacher_logits = self.teacher_logits.to(device, non_blocking=True)
        return self

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.input_ids = self.input_ids.pin_memory()
        self.targets = [target_id.pin_memory() for target_id in self.targets]
        if self.attention_masks is not None:
            self.attention_masks = self.attention_masks.pin_memory()
        if self.teacher_logits is not None:
            self.teacher_logits = self.teacher_logits.pin_memory()
        return self

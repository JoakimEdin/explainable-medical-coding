import gc
from collections import defaultdict
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Callable, Optional

import pandas as pd
import torch
from omegaconf import OmegaConf
from rich.pretty import pprint
from rich.progress import track
from torch.utils.data import DataLoader

from explainable_medical_coding.eval.metrics import MetricCollection
from explainable_medical_coding.trainer.callbacks import BaseCallback
from explainable_medical_coding.utils.datatypes import Lookups
from explainable_medical_coding.utils.decision_boundary import f1_score_db_tuning
from explainable_medical_coding.utils.settings import ID_COLUMN, TARGET_COLUMN


class Trainer:
    def __init__(
        self,
        config: OmegaConf,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloaders: dict[str, DataLoader],
        metric_collections: dict[str, MetricCollection],
        callbacks: list[BaseCallback],
        lookups: Lookups,
        loss_function: Callable,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        accumulate_grad_batches: int = 1,
    ) -> None:
        self.config = config
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.callbacks = callbacks
        self.device = "cpu"
        self.metric_collections = metric_collections
        self.lr_scheduler = lr_scheduler
        self.lookups = lookups
        self.accumulate_grad_batches = accumulate_grad_batches
        pprint(f"Accumulating gradients over {self.accumulate_grad_batches} batch(es).")
        self.validate_on_training_data = config.trainer.validate_on_training_data
        self.print_metrics = config.trainer.print_metrics
        self.epochs = config.trainer.epochs
        self.epoch = 0
        self.use_amp = config.trainer.use_amp
        self.threshold_tuning = config.trainer.threshold_tuning
        self.gradient_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.experiment_path = Path(mkdtemp())
        self.current_val_results: dict[str, dict[str, torch.Tensor]] = {}
        self.stop_training = False
        self.best_db = 0.5
        self.on_initialisation_end()

    def fit(self) -> None:
        """Train and validate the model."""
        try:
            self.save_configs()
            self.on_fit_begin()
            for _ in range(self.epoch, self.epochs):
                if self.stop_training:
                    break
                self.on_epoch_begin()
                self.train_one_epoch(self.epoch)
                if self.validate_on_training_data:
                    self.train_val(self.epoch, "train_val")
                self.val(self.epoch, "validation")
                self.on_epoch_end()
                self.epoch += 1
            self.on_fit_end()
            self.val(self.epoch, "validation", evaluating_best_model=True)
            self.val(self.epoch, "test", evaluating_best_model=True)

        except KeyboardInterrupt:
            pprint("Training interrupted by user. Stopping training")
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.on_end()

    def train_one_epoch(self, epoch: int) -> None:
        """Train the model for one epoch.

        Args:
            epoch (int): The current epoch.
        """
        self.model.train()
        self.on_train_begin()
        num_batches = len(self.dataloaders["train"])
        for batch_idx, batch in enumerate(
            track(self.dataloaders["train"], description=f"Epoch: {epoch} | Training")
        ):
            with torch.autocast(
                device_type="cuda", enabled=self.use_amp, dtype=torch.bfloat16
            ):
                batch = batch.to(self.device)
                y_probs, targets, loss = self.loss_function(
                    batch,
                    model=self.model,
                    scale=self.gradient_scaler.get_scale(),
                    epoch=epoch,
                )
                loss = loss / self.accumulate_grad_batches
            self.gradient_scaler.scale(loss).backward()
            if ((batch_idx + 1) % self.accumulate_grad_batches == 0) or (
                batch_idx + 1 == num_batches
            ):
                if self.config.trainer.clip_grad_norm:
                    self.gradient_scaler.unscale_(self.optimizer)
                    # torch.nn.utils.clip_grad_value_(norm=self.model.parameters(), clip_value=self.config.trainer.clip_value)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.trainer.clip_grad_norm
                    )
                self.gradient_scaler.step(self.optimizer)
                self.gradient_scaler.update()
                if self.lr_scheduler is not None:
                    if not isinstance(
                        self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        self.lr_scheduler.step()
                self.optimizer.zero_grad()
            self.update_metrics(
                y_probs=y_probs, targets=targets, loss=loss, split_name="train"
            )
        self.on_train_end(epoch)

    @torch.no_grad()
    def train_val(self, epoch, split_name: str = "train_val") -> None:
        """Validate on the training data. This is useful for testing for overfitting. Due to memory constraints, we donøt save the outputs.

        Args:
            epoch (_type_): _description_
            split_name (str, optional): _description_. Defaults to "train_val".
        """
        self.model.eval()
        self.on_val_begin()

        for batch in track(
            self.dataloaders[split_name],
            description=f"Epoch: {epoch} | Validating on training data",
        ):
            with torch.autocast(
                device_type="cuda", enabled=self.use_amp, dtype=torch.bfloat16
            ):
                y_probs, targets, loss = self.loss_function(
                    batch.to(self.device), model=self.model
                )
            self.update_metrics(
                y_probs=y_probs, targets=targets, loss=loss, split_name=split_name
            )
        self.on_val_end(split_name, epoch)

    @torch.no_grad()
    def val(
        self, epoch, split_name: str = "validation", evaluating_best_model: bool = False
    ) -> None:
        self.model.eval()
        self.on_val_begin()
        y_probs_list = []
        targets_list = []
        y_probs_cpu = []
        targets_cpu = []
        ids = []

        for idx, batch in enumerate(
            track(
                self.dataloaders[split_name],
                description=f"Epoch: {epoch} | Validating on {split_name}",
            )
        ):
            with torch.autocast(
                device_type="cuda", enabled=self.use_amp, dtype=torch.bfloat16
            ):
                y_probs, targets, loss = self.loss_function(
                    batch.to(self.device), model=self.model
                )
            self.update_metrics(
                y_probs=y_probs, targets=targets, loss=loss, split_name=split_name
            )
            y_probs_list.append(y_probs)
            targets_list.append(targets)
            ids.append(batch.ids)
            if idx % 1000 == 0:
                # move to cpu to save gpu memory
                y_probs_cpu.append(torch.cat(y_probs_list, dim=0).cpu())
                targets_cpu.append(torch.cat(targets_list, dim=0).cpu())
                y_probs_list = []
                targets_list = []
        y_probs_cpu.append(torch.cat(y_probs_list, dim=0).cpu())
        targets_cpu.append(torch.cat(targets_list, dim=0).cpu())

        y_probs = torch.cat(y_probs_cpu, dim=0)
        targets = torch.cat(targets_cpu, dim=0)
        ids = [item for sublist in ids for item in sublist]
        self.on_val_end(split_name, epoch, y_probs, targets, ids, evaluating_best_model)

    def update_metrics(
        self,
        y_probs: torch.Tensor,
        targets: torch.Tensor,
        loss: torch.Tensor,
        split_name: str,
    ) -> None:
        self.metric_collections[split_name].update(y_probs, targets, loss)

    def calculate_metrics(
        self,
        split_name: str,
        y_probs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        evaluating_best_model: bool = False,
    ) -> dict[str, dict[str, torch.Tensor]]:
        results_dict: dict[str, dict[str, Any]] = defaultdict(dict)
        if split_name == "validation":
            results_dict[split_name] = self.metric_collections[split_name].compute()
        else:
            results_dict[split_name] = self.metric_collections[split_name].compute(
                y_probs, targets
            )

        if self.threshold_tuning and split_name == "validation":
            best_result, best_db = f1_score_db_tuning(y_probs, targets)
            results_dict[split_name] |= {"f1_micro_tuned": best_result}
            if evaluating_best_model:
                pprint(f"Best threshold: {best_db}")
                pprint(f"Best result: {best_result}")
                self.metric_collections["test"].set_threshold(best_db)
            self.best_db = best_db
        return results_dict

    def reset_metric(self, split_name: str) -> None:
        self.metric_collections[split_name].reset_metrics()

    def reset_metrics(self) -> None:
        for split_name in self.metric_collections.keys():
            self.metric_collections[split_name].reset_metrics()

    def on_initialisation_end(self) -> None:
        for callback in self.callbacks:
            callback.on_initialisation_end(self)

    def on_fit_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_fit_begin(self)

    def on_fit_end(self) -> None:
        for callback in self.callbacks:
            callback.on_fit_end(self)

    def on_train_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self, epoch: int) -> None:
        results_dict = self.calculate_metrics(split_name="train")
        results_dict["lr"] = self.optimizer.param_groups[0]["lr"]
        self.log_dict(results_dict, epoch)
        for callback in self.callbacks:
            callback.on_train_end()

    def on_val_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_val_begin()

    def on_val_end(
        self,
        split_name: str,
        epoch: int,
        logits: torch.Tensor = None,
        targets: torch.Tensor = None,
        ids: Optional[list[int]] = None,
        evaluating_best_model: bool = False,
    ) -> None:
        results_dict = self.calculate_metrics(
            split_name=split_name,
            y_probs=logits,
            targets=targets,
            evaluating_best_model=evaluating_best_model,
        )
        self.current_val_results = results_dict
        self.log_dict(results_dict, epoch)
        for callback in self.callbacks:
            callback.on_val_end()

        if evaluating_best_model:
            self.save_predictions(
                split_name=split_name, logits=logits, targets=targets, ids=ids
            )

    def save_predictions(
        self,
        split_name: str = "test",
        logits: torch.Tensor = None,
        targets: torch.Tensor = None,
        ids: Optional[list[int]] = None,
    ):
        target_tokenizer = self.lookups.target_tokenizer
        code_names = target_tokenizer.target_names()
        logits = logits.numpy()
        df = pd.DataFrame(logits, columns=code_names)
        df[TARGET_COLUMN] = list(map(target_tokenizer.torch_one_hot_decoder, targets))
        df[ID_COLUMN] = ids
        df.to_feather(self.experiment_path / f"predictions_{split_name}.feather")

    def on_epoch_begin(self) -> None:
        self.reset_metrics()
        for callback in self.callbacks:
            callback.on_epoch_begin(self)

    def on_epoch_end(self) -> None:
        if self.lr_scheduler is not None:
            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.lr_scheduler.step(
                    self.current_val_results["validation"]["f1_micro"]
                )

        for callback in self.callbacks:
            callback.on_epoch_end(self)

    def on_batch_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_batch_begin()

    def on_batch_end(self) -> None:
        for callback in self.callbacks:
            callback.on_batch_end()

    def log_dict(
        self, nested_dict: dict[str, dict[str, torch.Tensor]], epoch: int
    ) -> None:
        if self.print_metrics:
            self.print(nested_dict)
        for callback in self.callbacks:
            callback.log_dict(nested_dict, epoch)

    def on_end(self) -> None:
        for callback in self.callbacks:
            callback.on_end()

    def print(self, nested_dict: dict[str, dict[str, Any]]) -> None:
        for split_name in nested_dict.keys():
            pprint(nested_dict[split_name])

    def to(self, device: str) -> "Trainer":
        self.model.to(device)
        for split_name in self.metric_collections.keys():
            self.metric_collections[split_name].to(device)
        self.device = device
        return self

    def save_checkpoint(self, file_name: str) -> None:
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.gradient_scaler.state_dict(),
            "epoch": self.epoch,
            "db": self.best_db,
            "num_classes": self.lookups.data_info["num_classes"],
        }
        torch.save(checkpoint, self.experiment_path / file_name)
        pprint("Saved checkpoint to {}".format(self.experiment_path / file_name))

    def load_checkpoint(self, file_name: str) -> None:
        checkpoint = torch.load(self.experiment_path / file_name)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.gradient_scaler.load_state_dict(checkpoint["scaler"])
        self.epoch = checkpoint["epoch"]
        self.best_db = checkpoint["db"]
        pprint("Loaded checkpoint from {}".format(self.experiment_path / file_name))

    def save_configs(self) -> None:
        self.lookups.target_tokenizer.save(
            self.experiment_path / "target_tokenizer.json"
        )
        OmegaConf.save(self.config, self.experiment_path / "config.yaml")

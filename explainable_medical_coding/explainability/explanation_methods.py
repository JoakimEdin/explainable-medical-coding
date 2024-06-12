import collections
import functools
from copy import deepcopy
from typing import Callable, Optional

import captum
import numpy as np
import torch
import transformers
from captum.attr._core.lrp import SUPPORTED_NON_LINEAR_LAYERS
from captum.attr._utils.lrp_rules import EpsilonRule, IdentityRule
from torch.utils.hooks import RemovableHandle

from explainable_medical_coding.explainability.helper_functions import (
    create_attention_mask,
    create_baseline_input,
    embedding_attributions_to_token_attributions,
    reshape_plm_icd_attributions,
)
from explainable_medical_coding.models.models import PLMICD, ModuleNames

Explainer = Callable[[torch.Tensor, torch.Tensor, str | torch.device], torch.Tensor]
SUPPORTED_NON_LINEAR_LAYERS.append(transformers.activations.GELUActivation)
PAD_ID = 1


def get_module(
    module: torch.nn.Module,
    module_name: str,
    layer_idx: int,
    model_layer_name: str = "roberta_encoder.encoder.layer",
) -> torch.nn.Module:
    """Get module from model.

    Args:
        module (nn.Module): Model
        module_name (str): Name of the module to get
        layer_idx (int): Layer index
        model_layer_name (str, optional): Name of the ModuleList containing all the transformer layers. Defaults to "roberta_encoder.encoder.layer".

    Returns:
        nn.Module: Module of interest
    """
    for sub_module_name in model_layer_name.split("."):
        module = getattr(module, sub_module_name)
    module = module[layer_idx]
    for sub_module_name in module_name.split("."):
        module = getattr(module, sub_module_name)
    return module


def get_modules(
    model: torch.nn.Module, layer_idx: int, module_names: ModuleNames
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    """Get modules specified in module names from model.

    Args:
        model (nn.Module): Model
        layer_idx (int): Layer index
        module_names (ModuleNames): Module names

    Returns:
        tuple[nn.Module, nn.Module, nn.Module, nn.Module]: LayerNorm 1, LayerNorm 2, dense to calculate attention values, dense to project the concatenated heads.
    """

    model_layer_name = module_names.model_layer_name
    ln_1 = get_module(model, module_names.ln_1, layer_idx, model_layer_name)
    ln_2 = get_module(model, module_names.ln_2, layer_idx, model_layer_name)
    dense_values = get_module(
        model, module_names.dense_values, layer_idx, model_layer_name
    )
    dense_heads = get_module(
        model, module_names.dense_heads, layer_idx, model_layer_name
    )

    return ln_1, ln_2, dense_values, dense_heads


class ALTI(torch.nn.Module):
    def __init__(self, model: PLMICD):
        super().__init__()
        self.model = model
        self.module_names: ModuleNames = model.module_names
        self.handles: dict[str, RemovableHandle] = {}
        self.func_inputs: dict[str, list[torch.Tensor]] = collections.defaultdict(list)
        self.func_outputs: dict[str, list[torch.Tensor]] = collections.defaultdict(list)

    def save_activation(
        self,
        name: str,
        module: torch.nn.Module,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
    ):
        self.func_inputs[name].append(inputs)
        self.func_outputs[name].append(outputs)

    def clean_hooks(self):
        for key in self.handles.keys():
            self.handles[key].remove()

    def register_hooks(self) -> None:
        for name, module in self.model.named_modules():
            self.handles[name] = module.register_forward_hook(
                functools.partial(self.save_activation, name)
            )

    def get_post_value_layer_states(self, layer_idx: int) -> torch.Tensor:
        return self.func_outputs[
            f"{self.module_names.model_layer_name}.{layer_idx}.{self.module_names.dense_values}"
        ][0]

    def change_value_dimensions(
        self, value: torch.Tensor, num_heads: int, head_size: int
    ) -> torch.Tensor:
        new_value_shape = value.size()[:-1] + (num_heads, head_size)
        value = value.view(*new_value_shape)
        return value.permute(0, 2, 1, 3)

    def get_pre_ln_states(self, layer_idx: int) -> torch.Tensor:
        return self.func_inputs[
            f"{self.module_names.model_layer_name}.{layer_idx}.{self.module_names.ln_1}"
        ][0][0]

    def get_post_ln_states(self, layer_idx: int) -> torch.Tensor:
        return self.func_outputs[
            f"{self.module_names.model_layer_name}.{layer_idx}.{self.module_names.ln_1}"
        ][0]

    @torch.no_grad()
    def l_transform(self, x, w_ln):
        """Computes mean and performs hadamard product with ln weight (w_ln) as a linear transformation."""
        ln_param_transf = torch.diag(w_ln)
        ln_mean_transf = torch.eye(w_ln.size(0)).to(w_ln.device) - 1 / w_ln.size(
            0
        ) * torch.ones_like(ln_param_transf).to(w_ln.device)

        out = torch.einsum(
            "... e , e f , f g -> ... g", x, ln_mean_transf, ln_param_transf
        )
        return out

    @torch.no_grad()
    def __call__(self, input_ids: torch.Tensor, attention_masks: torch.Tensor):
        self.register_hooks()
        (
            hidden_states,
            attentions,
            label_wise_attention,
        ) = self.model.get_encoder_attention_and_hidden_states(
            input_ids, attention_masks
        )
        attentions = (
            torch.stack(attentions).to(torch.float16).transpose(1, 0)
        )  # [batch_size*num_chunks, num_layers, num_heads, chunk_size, chunk_size]
        hidden_states = (
            torch.stack(hidden_states).to(torch.float16).transpose(1, 0)
        )  # [batch_size*num_chunks, num_layers+1, chunk_size, embedding_size]

        batch_size = input_ids.size(0)
        num_chunks = attentions.size(0) // batch_size
        num_layers = attentions.size(1)
        num_heads = attentions.size(2)
        chunk_size = attentions.size(3)
        embedding_size = hidden_states.size(-1)
        head_size = embedding_size // num_heads

        attentions = attentions.view(
            batch_size, num_chunks, num_layers, num_heads, chunk_size, chunk_size
        )
        hidden_states = hidden_states.view(
            batch_size, num_chunks, num_layers + 1, chunk_size, embedding_size
        )
        token_context = []
        for chunk_idx in range(num_chunks):
            chunk_token_context = None

            for layer_idx in range(num_layers):
                attentions_chunk_layer = attentions[:, chunk_idx, layer_idx]
                hidden_states_chunk_layer = hidden_states[:, chunk_idx, layer_idx]
                ln_1, ln_2, dense_values, dense_heads = get_modules(
                    self.model,
                    layer_idx,
                    self.module_names,
                )

                # [batch_size*num_chunks, chunk_size, embedding_size]
                post_value_states = self.get_post_value_layer_states(layer_idx)

                # [batch_size, chunk_size, embedding_size]
                post_value_states = post_value_states.view(
                    batch_size, num_chunks, chunk_size, embedding_size
                )[:, chunk_idx]

                # [batch_size, num_heads, chunk_size, embedding_size, head_size]
                post_value_states = self.change_value_dimensions(
                    post_value_states, num_heads, head_size
                )

                # [batch_size, chunk_size, embedding_size]
                pre_ln_states = self.get_pre_ln_states(layer_idx=layer_idx).view(
                    batch_size, num_chunks, chunk_size, embedding_size
                )[:, chunk_idx]

                # [embedding_size]
                dense_heads_bias = dense_heads.bias

                # [embedding_size, num_heads, head_size]
                dense_heads_weight = dense_heads.weight.view(
                    embedding_size, num_heads, head_size
                )

                # VW_O
                # (batch, num_heads, chunk_size, embedding_size)
                transformed_layer = torch.einsum(
                    "bhsv,dhv->bhsd", post_value_states, dense_heads_weight
                )
                del dense_heads_weight

                # AVW_O
                # (batch, num_heads, chunk_size, chunk_size, embedding_size)
                weighted_layer = torch.einsum(
                    "bhks,bhsd->bhksd", attentions_chunk_layer, transformed_layer
                )
                del transformed_layer

                # sum the weighted layer over all heads sum_h(AVW_O)
                summed_weighted_layer = weighted_layer.sum(
                    dim=1
                )  # (batch, chunk_size, chunk_size, embedding_size)
                del weighted_layer

                # Make residual matrix (batch, chunk_size, chunk_size, embedding_size)
                residual = torch.einsum(
                    "sk,bsd->bskd",
                    torch.eye(hidden_states_chunk_layer.shape[1]).to(
                        hidden_states_chunk_layer.device
                    ),
                    hidden_states_chunk_layer,
                )

                # AVW_O + residual vectors -> (batch, seq_len, seq_len, embed_dim)
                residual_weighted_layer = summed_weighted_layer + residual

                # LayerNorm 1
                ln_1_weight = ln_1.weight.data
                ln_1_eps = ln_1.eps
                ln_1_bias = ln_1.bias

                ln_std_coef = 1 / (pre_ln_states + ln_1_eps).std(
                    -1, unbiased=False
                ).view(1, -1, 1)

                # Transformed vectors
                # [batch_size, chunk_size, chunk_size, embedding_size]
                transformed_vectors = self.l_transform(
                    residual_weighted_layer, ln_1_weight
                )

                # Output vectors 1 per source token
                # [batch_size, chunk_size, embedding_size]
                attn_output = transformed_vectors.sum(dim=2)

                # Lb_O
                # [embedding_size]
                dense_heads_bias_term = self.l_transform(dense_heads_bias, ln_1_weight)

                # y_i = (sum(T_i(x_j)) + Lb_O)*ln_std_coef + ln_1_bias
                # [batch_size, chunk_size, embedding_size]
                resultant = (
                    attn_output + dense_heads_bias_term
                ) * ln_std_coef + ln_1_bias

                # T_i(x_j)
                # [batch_size, chunk_size, chunk_size, embedding_size]
                transformed_vectors_std = transformed_vectors * ln_std_coef.unsqueeze(
                    -1
                )

                # d_ij = ||T_i(x_j) - y_i||_1
                # [batch_size, chunk_size, chunk_size]
                distance_matrix = torch.nn.functional.pairwise_distance(
                    transformed_vectors_std, resultant.unsqueeze(2), p=1
                )

                # c_ij = max(0, ||y_i||_1 - d_ij)
                # [batch_size, chunk_size, chunk_size]
                contribution_matrix = torch.clip(
                    torch.norm(resultant, p=1, dim=-1).unsqueeze(1) - distance_matrix,
                    min=0,
                )

                # normalize contribution matrix
                # [batch_size, chunk_size, chunk_size]
                contribution_matrix = contribution_matrix / contribution_matrix.sum(
                    -1, keepdim=True
                )
                if chunk_token_context is None:
                    chunk_token_context = contribution_matrix
                else:
                    chunk_token_context = contribution_matrix @ chunk_token_context
                token_context.append(chunk_token_context)

        token_contribution_matrix = torch.zeros(
            batch_size,
            num_chunks * chunk_size,
            num_chunks * chunk_size,
            dtype=torch.float16,
            device=attentions.device,
        )
        for chunk_idx in range(num_chunks):
            token_contribution_matrix[
                :,
                chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size,
                chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size,
            ] = token_context[chunk_idx]

        self.clean_hooks()
        self.func_outputs.clear()
        self.func_inputs.clear()
        attributions = (
            label_wise_attention.to(torch.float16) @ token_contribution_matrix
        )
        return attributions / attributions.sum(-1, keepdim=True)


@torch.no_grad()
def get_occlusion_1_callable(
    model: PLMICD,
    baseline_token_id: int = 1,
    **kwargs,
) -> Explainer:
    def occlusion_1_callable(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        device: str | torch.device,
    ) -> torch.Tensor:
        input_ids = input_ids.to(device)
        sequence_length = input_ids.shape[1]
        num_classes = target_ids.shape[0]
        attributions = torch.zeros(sequence_length, num_classes)
        attention_mask = create_attention_mask(input_ids)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            y_pred = (
                model(input_ids, attention_mask)[:, target_ids]
                .detach()
                .cpu()
                .squeeze(0)
            )  # [num_classes]

        for idx in range(sequence_length):
            permuted_input_ids = input_ids.clone()
            permuted_input_ids[:, idx] = baseline_token_id
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                y_permute = (
                    model(permuted_input_ids, attention_mask)[:, target_ids]
                    .detach()
                    .cpu()
                    .squeeze(0)
                )  # [num_classes]
            attributions[idx] = y_pred - y_permute

        attributions = torch.abs(attributions)
        return attributions / torch.norm(attributions, p=1)

    return occlusion_1_callable


@torch.no_grad()
def get_attention_rollout_callable(
    model: PLMICD,
    **kwargs,
) -> Explainer:
    """Get a callable LAAT explainer

    Args:
        model (PLMICD): Model to explain

    Returns:
        Explainer: Callable LAAT explainer
    """

    def attention_rollout_callable(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        device: str | torch.device,
    ) -> torch.Tensor:
        input_ids = input_ids.to(device)
        sequence_length = input_ids.shape[1]
        attention_mask = create_attention_mask(input_ids)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            label_attentions = model.attention_rollout(input_ids, attention_mask)
        label_attentions = label_attentions.squeeze(
            0
        ).T.detach()  # [sequence_length+padding, num_classes]
        label_attentions = label_attentions[:sequence_length, target_ids]
        return label_attentions.cpu()

    return attention_rollout_callable


@torch.no_grad()
def get_alti_callable(
    model: PLMICD,
    **kwargs,
) -> Explainer:
    """Get a callable ALTI explainer

    Args:
        model (PLMICD): Model to explain

    Returns:
        Explainer: Callable LAAT explainer
    """
    alti = ALTI(model)

    def alti_callable(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        device: str | torch.device,
    ) -> torch.Tensor:
        input_ids = input_ids.to(device)
        sequence_length = input_ids.shape[1]
        attention_mask = create_attention_mask(input_ids)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            label_attentions = alti(input_ids, attention_mask)
        label_attentions = torch.softmax(label_attentions, dim=-1)
        label_attentions = label_attentions.squeeze(
            0
        ).T.detach()  # [sequence_length+padding, num_classes]
        label_attentions = label_attentions[:sequence_length, target_ids]
        return label_attentions.cpu()

    return alti_callable


@torch.no_grad()
def get_laat_callable(
    model: PLMICD,
    **kwargs,
) -> Explainer:
    """Get a callable LAAT explainer

    Args:
        model (PLMICD): Model to explain

    Returns:
        Explainer: Callable LAAT explainer
    """

    def laat_callable(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        device: str | torch.device,
    ) -> torch.Tensor:
        input_ids = input_ids.to(device)
        sequence_length = input_ids.shape[1]
        attention_mask = create_attention_mask(input_ids)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            _, label_attentions = model(
                input_ids, attention_mask, output_attentions=True
            )
        label_attentions = torch.softmax(label_attentions, dim=-1)
        label_attentions = label_attentions.squeeze(
            0
        ).T.detach()  # [sequence_length+padding, num_classes]
        label_attentions = label_attentions[:sequence_length, target_ids]
        return label_attentions.cpu()

    return laat_callable


@torch.no_grad()
def get_grad_attention_callable(
    model: PLMICD,
    **kwargs,
) -> Explainer:
    """Get a callable LAAT explainer

    Args:
        model (PLMICD): Model to explain

    Returns:
        Explainer: Callable LAAT explainer
    """
    attention_dict = {}

    def forward(_chunked_embeddings: torch.Tensor, attention_masks: torch.Tensor):
        logits, attention = model.forward_embedding_input(
            _chunked_embeddings, attention_masks=attention_masks, output_attention=True
        )
        attention_dict["attention"] = attention
        return logits

    def grad_attention_callable(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        device: str | torch.device,
    ) -> np.ndarray:
        input_ids = input_ids.to(device)
        attention_mask = create_attention_mask(input_ids)
        chunked_embeddings = model.get_chunked_embedding(input_ids)
        sequence_length = input_ids.shape[1]
        num_classes = target_ids.shape[0]

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            input_gradient = torch.autograd.functional.jacobian(
                lambda _chunked_embeddings: forward(
                    _chunked_embeddings, attention_masks=attention_mask
                )[0, target_ids],
                chunked_embeddings,
            ).detach()

        # attn_gradient = input_gradients_dict["gradient"].detach().cpu()
        gradient_x_input = (
            chunked_embeddings.repeat(num_classes, 1, 1, 1) * input_gradient
        )
        gradient_x_input = gradient_x_input.view(
            num_classes,
            gradient_x_input.shape[1] * gradient_x_input.shape[2],
            gradient_x_input.shape[-1],
        )

        gradient_x_input = gradient_x_input[:, :sequence_length].transpose(0, 1)
        gradient_x_input = embedding_attributions_to_token_attributions(
            gradient_x_input
        )

        label_attentions = attention_dict["attention"].detach()
        label_attentions = torch.softmax(label_attentions, dim=-1)
        label_attentions = label_attentions.squeeze(
            0
        ).T.detach()  # [sequence_length+padding, num_classes]
        label_attentions = label_attentions[:sequence_length, target_ids]

        attributions = label_attentions * gradient_x_input
        return (attributions / torch.norm(attributions, p=1)).cpu()

    return grad_attention_callable


def get_atgrad_attention_callable(
    model: PLMICD,
    **kwargs,
) -> Explainer:
    """Get a callable LAAT explainer

    Args:
        model (PLMICD): Model to explain

    Returns:
        Explainer: Callable LAAT explainer
    """
    attn_gradients_dict = {}

    def save_attn_gradients(
        gradients: torch.Tensor,
    ):
        attn_gradients_dict["gradient"] = gradients

    def atgrad_attention_callable(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        device: str | torch.device,
    ) -> np.ndarray:
        input_ids = input_ids.to(device)
        attention_mask = create_attention_mask(input_ids)
        sequence_length = input_ids.shape[1]
        num_classes = target_ids.shape[0]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            logits, label_attentions = model(
                input_ids,
                attention_mask,
                output_attentions=True,
                attn_grad_hook_fn=save_attn_gradients,
            )

        label_attentions = torch.softmax(label_attentions.detach(), dim=-1)
        label_attentions = label_attentions.squeeze(
            0
        ).T.detach()  # [sequence_length+padding, num_classes]
        label_attentions = label_attentions[:sequence_length].detach().cpu()

        class_attributions = torch.zeros(sequence_length, num_classes)
        for idx, target in enumerate(target_ids):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits[:, target].backward(retain_graph=True)
                model.zero_grad()
                attn_gradient = attn_gradients_dict["gradient"].detach().cpu()
                attn_gradient = attn_gradient.squeeze(
                    0
                ).T  # [sequence_length+padding, num_classes]
                attn_gradient = attn_gradient[:sequence_length]
                attributions = (
                    torch.abs(attn_gradient[:, target]) * label_attentions[:, target]
                )
                class_attributions[:, idx] = attributions / torch.norm(
                    attributions, p=1
                )

        return class_attributions

    return atgrad_attention_callable


def get_deeplift_callable(
    model: PLMICD,
    baseline_token_id: int = 1,
    cls_token_id: int = 0,
    eos_token_id: int = 2,
    **kwargs,
) -> Explainer:
    """Get a callable DeepLift explainer

    Args:
        model (PLMICD): Model to explain
        baseline_token_id (int, optional): Baseline token id. Defaults to 50_000.
        cls_token_id (int, optional): CLS token id. Defaults to 0.
        eos_token_id (int, optional): SEP token id. Defaults to 2.

    Returns:
        Explainer: Callable DeepLift explainer
    """
    explainer = captum.attr.LayerDeepLift(model, model.roberta_encoder.embeddings)

    def deeplift_callable(
        input_ids: torch.Tensor,
        device: str | torch.device,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate DeepLift attributions for each class in target_ids.

        Args:
            inputs (torch.Tensor): Input token ids [batch_size, sequence_length]
            device (str | torch.device): Device to use
            target_ids (torch.Tensor): Target token ids [num_classes]

        Returns:
            torch.Tensor: Attributions [sequence_length, num_classes]
        """
        sequence_length = input_ids.shape[1]
        attention_mask = input_ids.to(device)
        attention_mask = create_attention_mask(input_ids)
        baseline = create_baseline_input(
            input_ids,
            baseline_token_id=baseline_token_id,
            cls_token_id=cls_token_id,
            eos_token_id=eos_token_id,
        )

        num_classes = target_ids.shape[0]
        class_attributions = torch.zeros(sequence_length, num_classes)
        for idx, target in enumerate(target_ids):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                attributions = explainer.attribute(
                    input_ids,
                    baselines=baseline,
                    additional_forward_args=attention_mask,
                    target=target,
                )
            attributions = reshape_plm_icd_attributions(attributions, input_ids)
            attributions = embedding_attributions_to_token_attributions(attributions)
            class_attributions[:, idx] = attributions.detach().cpu()
        return class_attributions / class_attributions.sum(0)

    return deeplift_callable


def get_gradient_x_input_callable(
    model: PLMICD,
    **kwargs,
) -> Explainer:
    """Get a callable Gradient x Input explainer

    Args:
        model (PLMICD): Model to explain
        baseline_token_id (int, optional): Baseline token id. Defaults to 0.
        cls_token_id (int, optional): CLS token id. Defaults to 0.
        eos_token_id (int, optional): SEP token id. Defaults to 2.

    Returns:
        Explainer: Callable Gradient x Input explainer
    """
    explainer = captum.attr.LayerGradientXActivation(
        model, model.roberta_encoder.embeddings, multiply_by_inputs=True
    )

    def gradients_x_input_callable(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        device: str | torch.device,
    ) -> np.ndarray:
        input_ids = input_ids.to(device)
        attention_mask = create_attention_mask(input_ids)
        sequence_length = input_ids.shape[1]
        num_classes = target_ids.shape[0]
        class_attributions = torch.zeros(sequence_length, num_classes)
        for idx, target in enumerate(target_ids):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                attributions = explainer.attribute(
                    input_ids,
                    additional_forward_args=attention_mask,
                    target=target,
                )
            attributions = reshape_plm_icd_attributions(attributions, input_ids)
            attributions = embedding_attributions_to_token_attributions(attributions)
            class_attributions[:, idx] = attributions.detach().cpu()
        return class_attributions / class_attributions.sum(0)

    return gradients_x_input_callable


def get_integrated_gradient_callable(
    model: PLMICD,
    baseline_token_id: int = 1,
    cls_token_id: int = 0,
    eos_token_id: int = 2,
    batch_size: int = 16,
    **kwargs,
) -> Explainer:
    """Get a callable Integrated Gradients explainer

    Args:
        model (PLMICD): Model to explain
        baseline_token_id (int, optional): Baseline token id. Defaults to 0.
        cls_token_id (int, optional): CLS token id. Defaults to 0.
        eos_token_id (int, optional): EOS token id. Defaults to 2.

    Returns:
        Explainer: Callable Integrated Gradients explainer
    """
    explainer = captum.attr.LayerIntegratedGradients(
        model, model.roberta_encoder.embeddings, multiply_by_inputs=False
    )

    def integrated_gradients_callable(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        device: str | torch.device,
    ) -> np.ndarray:
        input_ids = input_ids.to(device)
        attention_mask = create_attention_mask(input_ids)
        baseline = create_baseline_input(
            input_ids,
            baseline_token_id=baseline_token_id,
            cls_token_id=cls_token_id,
            eos_token_id=eos_token_id,
        )
        sequence_length = input_ids.shape[1]
        num_classes = target_ids.shape[0]
        class_attributions = torch.zeros(sequence_length, num_classes)
        for idx, target in enumerate(target_ids):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                attributions = explainer.attribute(
                    input_ids,
                    baselines=baseline,
                    additional_forward_args=attention_mask,
                    target=target,
                    internal_batch_size=batch_size,
                )
            attributions = reshape_plm_icd_attributions(attributions, input_ids)
            attributions = embedding_attributions_to_token_attributions(attributions)
            class_attributions[:, idx] = attributions.detach().cpu()
        return class_attributions / class_attributions.sum(0)

    return integrated_gradients_callable


def get_kernelshap_callable(
    model: PLMICD,
    baseline_token_id: int = 1,
    cls_token_id: int = 0,
    eos_token_id: int = 2,
    sample_ratio: int = 3,
    **kwargs,
) -> Explainer:
    """Get a callable Kernel Shap explainer

    Args:
        model (PLMICD): Model to explain
        baseline_token_id (int, optional): Baseline token id. Defaults to 0.
        cls_token_id (int, optional): CLS token id. Defaults to 0.
        eos_token_id (int, optional): SEP token id. Defaults to 2.

    Returns:
        Explainer: Callable Integrated Gradients explainer
    """
    explainer = captum.attr.KernelShap(model)

    @torch.no_grad()
    def kernelshap_callable(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        device: str | torch.device,
        feature_mask: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        input_ids = input_ids.to(device)
        attention_mask = create_attention_mask(input_ids)
        baseline = create_baseline_input(
            input_ids,
            baseline_token_id=baseline_token_id,
            cls_token_id=cls_token_id,
            eos_token_id=eos_token_id,
        )

        sequence_length = input_ids.shape[1]
        num_classes = target_ids.shape[0]
        class_attributions = torch.zeros(sequence_length, num_classes)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            for idx, target in enumerate(target_ids):
                attributions = explainer.attribute(
                    input_ids,
                    baselines=baseline,
                    additional_forward_args=attention_mask.unsqueeze(0),
                    n_samples=sample_ratio * sequence_length,
                    target=target,
                    feature_mask=feature_mask,
                )
                class_attributions[:, idx] = attributions.squeeze().detach().cpu()
        class_attributions = torch.abs(class_attributions)
        return class_attributions / torch.norm(class_attributions, p=1)

    return kernelshap_callable

@torch.no_grad()
def get_lime_callable(
    model: PLMICD,
    baseline_token_id: int = 1,
    cls_token_id: int = 0,
    eos_token_id: int = 2,
    sample_ratio: int = 3,
    **kwargs,
) -> Explainer:
    """Get a callable Kernel Shap explainer

    Args:
        model (PLMICD): Model to explain
        baseline_token_id (int, optional): Baseline token id. Defaults to 0.
        cls_token_id (int, optional): CLS token id. Defaults to 0.
        eos_token_id (int, optional): SEP token id. Defaults to 2.

    Returns:
        Explainer: Callable Integrated Gradients explainer
    """
    explainer = captum.attr.Lime(model)

    @torch.no_grad()
    def lime_callable(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        device: str | torch.device,
    ) -> np.ndarray:
        input_ids = input_ids.to(device)
        attention_mask = create_attention_mask(input_ids)
        baseline = create_baseline_input(
            input_ids,
            baseline_token_id=baseline_token_id,
            cls_token_id=cls_token_id,
            eos_token_id=eos_token_id,
        )
        sequence_length = input_ids.shape[1]
        num_classes = target_ids.shape[0]
        class_attributions = torch.zeros(sequence_length, num_classes)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            for idx, target in enumerate(target_ids):
                attributions = explainer.attribute(
                    input_ids,
                    baselines=baseline,
                    additional_forward_args=attention_mask.unsqueeze(0),
                    n_samples=sample_ratio * sequence_length,
                    target=target,
                )
                class_attributions[:, idx] = attributions.squeeze().detach().cpu()
        class_attributions = torch.abs(class_attributions)
        return class_attributions / torch.norm(class_attributions, p=1)

    return lime_callable


def get_random_baseline_callable(**kwargs) -> Explainer:
    def random_baseline_callable(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        device: str | torch.device,
    ) -> torch.Tensor:
        sequence_length = input_ids.shape[1]
        num_classes = target_ids.shape[0]
        attributions = torch.abs(torch.randn((sequence_length, num_classes)))
        return attributions / attributions.sum(0, keepdim=True)

    return random_baseline_callable

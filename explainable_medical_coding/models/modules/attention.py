from typing import Optional, Callable

import torch
import torch.nn as nn


class LabelAttention(nn.Module):
    def __init__(self, input_size: int, projection_size: int, num_classes: int):
        super().__init__()
        self.first_linear = nn.Linear(input_size, projection_size, bias=False)
        self.second_linear = nn.Linear(projection_size, num_classes, bias=False)
        self.third_linear = nn.Linear(input_size, num_classes)
        self._init_weights(mean=0.0, std=0.03)

    def forward(
        self,
        x: torch.Tensor,
        output_attention: bool = False,
        attention_masks: Optional[torch.Tensor] = None,
        attn_grad_hook_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """LAAT attention mechanism

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        weights = torch.tanh(self.first_linear(x))  # [batch_size, seq_len, proj_size]
        att_weights = self.second_linear(weights)  # [batch_size, seq_len, num_classes]
        att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1, 2)
        weighted_output = att_weights @ x  # [batch_size, num_classes, input_size]
        output = (
            self.third_linear.weight.mul(weighted_output)
            .sum(dim=2)
            .add(self.third_linear.bias)
        )  # [batch_size, num_classes]
        if output_attention:
            return output, att_weights
        return output

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        torch.nn.init.normal_(self.first_linear.weight, mean, std)
        torch.nn.init.normal_(self.second_linear.weight, mean, std)
        torch.nn.init.normal_(self.third_linear.weight, mean, std)


class LabelCrossAttention(nn.Module):
    def __init__(self, input_size: int, num_classes: int, scale: float = 1.0):
        super().__init__()
        self.weights_k = nn.Linear(input_size, input_size, bias=False)
        self.label_representations = torch.nn.Parameter(
            torch.rand(num_classes, input_size), requires_grad=True
        )
        self.weights_v = nn.Linear(input_size, input_size)
        self.output_linear = nn.Linear(input_size, 1)
        self.layernorm = nn.LayerNorm(input_size)
        self.num_classes = num_classes
        self.scale = scale
        self._init_weights(mean=0.0, std=0.03)

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        output_attention: bool = False,
        attn_grad_hook_fn: Optional[Callable] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Label Cross Attention mechanism

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """

        V = self.weights_v(x)
        K = self.weights_k(x)
        Q = self.label_representations

        att_weights = Q.matmul(K.transpose(1, 2))

        # replace nan with max value of float16
        # att_weights = torch.where(
        #     torch.isnan(att_weights), torch.tensor(30000), att_weights
        # )
        if attention_masks is not None:
            # pad attention masks with 0 such that it has the same sequence lenght as x
            attention_masks = torch.nn.functional.pad(
                attention_masks, (0, x.size(1) - attention_masks.size(1)), value=0
            )
            attention_masks = attention_masks.to(torch.bool)
            # repeat attention masks for each class
            attention_masks = attention_masks.unsqueeze(1).repeat(
                1, self.num_classes, 1
            )
            attention_masks = attention_masks.masked_fill_(
                attention_masks.logical_not(), float("-inf")
            )
            att_weights += attention_masks

        attention = torch.softmax(
            att_weights / self.scale, dim=2
        )  # [batch_size, num_classes, seq_len]
        if attn_grad_hook_fn is not None:
            attention.register_hook(attn_grad_hook_fn)

        y = attention @ V  # [batch_size, num_classes, input_size]

        y = self.layernorm(y)

        output = (
            self.output_linear.weight.mul(y)
            .sum(dim=2)
            .add(self.output_linear.bias)  # [batch_size, num_classes]
        )

        if output_attention:
            return output, att_weights

        return output

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        self.weights_k.weight = torch.nn.init.normal_(self.weights_k.weight, mean, std)
        self.weights_v.weight = torch.nn.init.normal_(self.weights_v.weight, mean, std)
        self.label_representations = torch.nn.init.normal_(
            self.label_representations, mean, std
        )
        self.output_linear.weight = torch.nn.init.normal_(
            self.output_linear.weight, mean, std
        )


class InputMasker(nn.Module):
    def __init__(self, input_size: int, scale: float = 1.0):
        super().__init__()
        self.weights_k = nn.Linear(input_size, input_size)
        self.weights_q = nn.Linear(input_size, input_size)
        self.weights_v = nn.Linear(input_size, input_size)
        self.output_linear = nn.Linear(input_size, 1)
        self.layernorm = nn.LayerNorm(input_size)
        self.scale = scale
        self._init_weights(mean=0.0, std=0.03)

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        output_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Label Cross Attention mechanism

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """

        V = self.weights_v(x)
        K = self.weights_k(x)
        Q = self.weights_q(x)

        att_weights = Q.matmul(K.transpose(1, 2))

        # replace nan with max value of float16
        # att_weights = torch.where(
        #     torch.isnan(att_weights), torch.tensor(30000), att_weights
        # )
        if attention_masks is not None:
            # pad attention masks with 0 such that it has the same sequence lenght as x
            attention_masks = torch.nn.functional.pad(
                attention_masks, (0, x.size(1) - attention_masks.size(1)), value=0
            )
            attention_masks = attention_masks.to(torch.bool)

            # repeat attention masks for each token
            attention_masks = attention_masks.unsqueeze(1).repeat(1, x.size(1), 1)

            attention_masks = attention_masks.masked_fill_(
                attention_masks.logical_not(), float("-inf")
            )
            att_weights += attention_masks

        attention = torch.softmax(
            att_weights / self.scale, dim=2
        )  # [batch_size, num_classes, seq_len]

        y = attention @ V  # [batch_size, num_classes, input_size]

        y = self.layernorm(y)

        output = (
            self.output_linear.weight.mul(y)
            .sum(dim=2)
            .add(self.output_linear.bias)  # [batch_size, num_classes]
        )

        if output_attention:
            return output, att_weights

        return output

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        self.weights_k.weight = torch.nn.init.normal_(self.weights_k.weight, mean, std)
        self.weights_v.weight = torch.nn.init.normal_(self.weights_v.weight, mean, std)
        self.weights_q.weight = torch.nn.init.normal_(self.weights_q.weight, mean, std)
        self.output_linear.weight = torch.nn.init.normal_(
            self.output_linear.weight, mean, std
        )

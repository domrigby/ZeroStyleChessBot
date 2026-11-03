from torch import nn
import torch

from typing import List, Optional, Sequence, Tuple


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, num_repeats: int = 2, activation_period: int = 2) -> None:

        super().__init__()

        convs_list: List[nn.Module] = []

        for idx in range(num_repeats):
            convs_list.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            convs_list.append(nn.BatchNorm2d(in_channels))
            if idx % activation_period == 0:
                convs_list.append(nn.LeakyReLU())

        self.conv_block = nn.Sequential(*convs_list)
        self.out_act = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Perform convolutions
        out = self.conv_block(x)

        # Skip connection
        out = out + x
        out = self.out_act(out)

        return out


class ChessNet(nn.Module):
    def __init__(
        self,
        input_size: Sequence[int],
        output_size: Sequence[int],
        num_filters: int = 64,
        num_repeats: int = 6,
        num_policy_head_blocks: int = 3,
        num_value_head_blocks: int = 2,
        policy_loss_factor: float = 1.0,
        value_loss_factor: float = 0.5,
    ) -> None:

        # Initialise nn Module
        super().__init__()

        # Control inputs and outputs
        self.input_size = input_size
        self.output_size = output_size

        # Control convolution parameters
        self.num_filters = num_filters
        self.num_repeats = num_repeats
        self.num_policy_head_blocks = num_policy_head_blocks
        self.num_value_head_blocks = num_value_head_blocks

        # Initialise the network
        self._build_network()

        # Send to CUDA device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

        self.value_loss = nn.MSELoss()
        self.policy_loss = nn.CrossEntropyLoss()

        # Set the loss balancing factors
        self.pol_alpha: float = policy_loss_factor
        self.val_beta: float = value_loss_factor

    def _build_network(self) -> None:

        self.activation = nn.LeakyReLU()
        self.input_conv = nn.Conv2d(self.input_size[0], self.num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_filters)

        conv_block_repeats: List[ConvBlock] = []
        for _ in range(self.num_repeats):
            conv_block_repeats.append(ConvBlock(self.num_filters))

        self.conv_blocks = nn.Sequential(*conv_block_repeats)

        self.policy_head = nn.Sequential(
            *[ConvBlock(self.num_filters) for _ in range(self.num_policy_head_blocks)],
            nn.Conv2d(self.num_filters, self.output_size[0], kernel_size=3, padding=1),
        )

        self.value_head = nn.Sequential(
            *[ConvBlock(self.num_filters) for _ in range(self.num_value_head_blocks)],
            nn.Conv2d(self.num_filters, 4, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(4 * 64, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(
        self, x: torch.Tensor, legal_move_mask: Optional[torch.Tensor] = None, infering: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x = x.to(self.device)
        if legal_move_mask is not None:
            legal_move_mask = legal_move_mask.to(self.device)

        x = self.input_conv(x)
        x = self.bn1(x)

        x = self.activation(x)

        x = self.conv_blocks(x)

        value = self.value_head(x)

        policy = self.policy_head(x)

        size_pre_flat = policy.size()

        if legal_move_mask is not None:
            # finfo.min rather than -inf: a 0 target times log_softmax(-inf) would make the cross-entropy NaN
            policy = torch.where(legal_move_mask == 1, policy, torch.finfo(policy.dtype).min)

        if infering:
            # One distribution over all moves. A per-square softmax (dim=1) with the masked logits would give every
            # square holding a legal move an equal share, whatever the network's logits
            policy = torch.nn.functional.softmax(policy.view(policy.size(0), -1), dim=1).view(size_pre_flat)

        return value, policy

    def loss_function(
        self,
        input_tensor: torch.Tensor,
        target: Tuple[torch.Tensor, torch.Tensor],
        legal_move_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        target_value, target_policy = target
        target_value, target_policy = (
            target_value.to(self.device, non_blocking=True),
            target_policy.to(self.device, non_blocking=True),
        )

        # Predicted value
        predicted_value, predicted_policy = self(input_tensor, legal_move_mask)

        if target_value.ndim == 1:
            target_value = target_value.unsqueeze(-1)

        # Calculate losses
        value_loss = self.value_loss(predicted_value, target_value)
        policy_loss = self.policy_loss(predicted_policy, target_policy)

        total_loss = self.val_beta * value_loss + self.pol_alpha * policy_loss

        return total_loss, value_loss, policy_loss

    def save_network(self, filename: str) -> None:
        torch.save(self.state_dict(), filename)

    def load_network(self, filename: str) -> None:
        self.load_state_dict(torch.load(filename))

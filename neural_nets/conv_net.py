from neural_nets.generic_net import GenericNet, ConvBlock, check_input
from torch import nn
import torch

from typing import List

class ChessNet(GenericNet):

    def __init__(self, *args, num_filters: int = 64, num_repeats: int= 6, policy_loss_factor: float = 1.,
                 value_loss_factor: float = .5, **kwargs):

        # Control convolution parameters
        self.num_filters = num_filters
        self.num_repeats = num_repeats

        super().__init__(*args, **kwargs)

        self.value_loss = nn.MSELoss()
        self.policy_loss = nn.CrossEntropyLoss()

        # Set the loss balancing factors
        self.pol_alpha: float = policy_loss_factor
        self.val_beta: float = value_loss_factor

    def _build_network(self):

        self.activation = nn.LeakyReLU()
        self.input_conv = nn.Conv2d(self.input_size[0], self.num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_filters)

        conv_block_repeats: List[ConvBlock] = []
        for _ in range(self.num_repeats):
            conv_block_repeats.append(ConvBlock(self.num_filters))

        self.conv_blocks = nn.Sequential(*conv_block_repeats)

        self.policy_head = nn.Sequential(ConvBlock(self.num_filters),
                                         ConvBlock(self.num_filters),
                                         ConvBlock(self.num_filters),
                                         nn.Conv2d(self.num_filters, self.output_size[0], kernel_size=3, padding=1))

        self.value_head = nn.Sequential(ConvBlock(self.num_filters), ConvBlock(self.num_filters),
                                     nn.Conv2d(self.num_filters, 4, kernel_size=3, padding=1),
                                     nn.LeakyReLU(),
                                     nn.Flatten(),
                                     nn.Linear(4*64, 256),
                                     nn.LeakyReLU(),
                                     nn.Dropout(p=0.2),
                                     nn.BatchNorm1d(256),
                                     nn.Linear(256, 1),
                                     nn.Tanh())

    @check_input
    def forward(self, x: torch.tensor, legal_move_mask: torch.tensor = None, infering: bool = False):

        x = self.input_conv(x)
        x = self.bn1(x)

        x = self.activation(x)

        x = self.conv_blocks(x)

        value = self.value_head(x)

        policy = self.policy_head(x)

        size_pre_flat = policy.size()

        policy = policy.view(policy.size(0), -1)

        policy = policy.view(size_pre_flat)

        if legal_move_mask is not None:
            policy = torch.where(legal_move_mask == 1, policy, 0)

        if infering:
            policy = torch.nn.functional.softmax(policy, dim=1)

        return value, policy

    def loss_function(self, input_tensor: torch.tensor, target: tuple, legal_move_mask: torch.tensor = None, training: bool = True):

        target_value, target_policy = target
        target_value, target_policy = target_value.to(self.device, non_blocking=True), target_policy.to(self.device, non_blocking=True)

        # Predicted value
        predicted_value, predicted_policy = self(input_tensor, legal_move_mask)

        if target_value.ndim == 1:
            target_value = target_value.unsqueeze(-1)

        # Calculate losses
        value_loss = self.value_loss(predicted_value, target_value)
        policy_loss = self.policy_loss(predicted_policy, target_policy)

        total_loss = self.val_beta * value_loss + self.pol_alpha * policy_loss

        return total_loss, value_loss, policy_loss




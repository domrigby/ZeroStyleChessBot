import torch
from torch import nn
from torch.optim import Adam

class GenericNet(nn.Module):

    def __init__(self, init_lr: float = 0.001):

        # Initialise the network
        self._build_network()

        self.optimiser = Adam(self.parameters(), lr=init_lr)


    def _build_network(self):
        """
        Abstract method to build neural network. Create the network parameters in you init
        :return:
        """
        raise NotImplementedError

    def loss_function(self, input: torch.tensor, target: torch.tensor):
        raise NotImplementedError

class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, num_repeats: int = 1):

        convs_list = []

        for _ in range(num_repeats):
            convs_list.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            convs_list.append(nn.BatchNorm2d(in_channels))
            convs_list.append(nn.LeakyReLU())

        self.conv_block = nn.Sequential(*convs_list)
        self.out_act = nn.LeakyRelu()

    def forward(self, x: torch.tensor):

        # Perform convolutions
        out = self.conv(x)

        # Skip connection
        out = out + x
        out = self.out_act(out)

        return out


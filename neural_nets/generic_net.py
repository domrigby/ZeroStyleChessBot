import torch
from torch import nn
from torch.optim import Adam
from torch.optim import lr_scheduler
import torch
from functools import wraps
import os

def check_input(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Ensure `self` has a `device` attribute
        if not hasattr(self, 'device'):
            raise AttributeError(f"{self.__class__.__name__} must have a 'device' attribute to use @ensure_device_method.")

        device = self.device

        # Move all tensor arguments to the correct device
        args = tuple(arg.to(device) if isinstance(arg, torch.Tensor) and arg.device != device else arg for arg in args)
        kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) and v.device != device else v for k, v in kwargs.items()}

        return func(self, *args, **kwargs)
    return wrapper

class GenericNet(nn.Module):

    def __init__(self, init_lr: float = 0.001, save_dir: str = 'networks'):

        # Initialise nn Module
        super().__init__()

        self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        # Initialise the network
        self._build_network()

        self.optimiser = Adam(self.parameters(), lr=init_lr) #, weight_decay=0.001)
        # Initialize the scheduler
        self.scheduler = lr_scheduler.StepLR(self.optimiser, step_size=1000, gamma=0.97)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.to(self.device)


    def _build_network(self):
        """
        Abstract method to build neural network. Create the network parameters in you init
        :return:
        """
        raise NotImplementedError

    def loss_function(self, input: torch.tensor, target: torch.tensor):
        raise NotImplementedError

    def save_network(self, filename: str = None):
        if filename is None:
            filename = os.path.join(self.save_dir, self.__class__.__name__) + '.pt'
        torch.save(self.state_dict(), filename)

    def load_network(self, filename: str):
        if filename is None:
            filename = os.path.join(self.save_dir, self.__class__.__name__) + '.pt'
        self.load_state_dict(torch.load(filename))

class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, num_repeats: int = 1):

        super().__init__()

        convs_list = []

        for _ in range(num_repeats):
            convs_list.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            convs_list.append(nn.LeakyReLU())
            convs_list.append(nn.BatchNorm2d(in_channels))

        self.conv_block = nn.Sequential(*convs_list)
        self.out_act = nn.LeakyReLU()

    def forward(self, x: torch.tensor):

        # Perform convolutions
        out = self.conv_block(x)

        # Skip connection
        out = out + x
        out = self.out_act(out)

        return out


from neural_nets.generic_net import GenericNet, ConvBlock
from torch import nn

class ChessNet(GenericNet):

    def __init__(self, input_size: tuple, output_size: tuple, *args, num_filters: int = 64, num_repeats: int= 6,
                 **kwargs):

        # Control inputs and outputs
        self.input_size = input_size
        self.output_size = output_size

        # Control convolution parameters
        self.num_filters = num_filters
        self.num_repeats = num_repeats

        super().__init__(*args, **kwargs)

    def _build_network(self):

        self.activation = nn.LeakyReLU()
        self.input_conv = nn.Conv2D(self.input_size[-1], self.num_filters, kernel_size=3, padding=1)

        conv_block_repeats = []
        for _ in range(self.num_repeats):
            conv_block_repeats.append(ConvBlock(self.num_filters))

        self.conv_blocks = nn.Sequential(*conv_block_repeats)

        self.output_conv = nn.Conv2D(self.num_filters, self.output_size[-1], kernel_size=3, padding=1)

        self.output_act = nn.Softmax(dim=1)  # Softmax for multi-class classification

    def forward(self, x: tensor, legal_moves: list[bool] = None):

        x = self.input_conv(x)
        x = self.activation(x)

        x = self.conv_blocks(x)
        x = self.output_conv(x)

        x = self.output_act(x)

        return x




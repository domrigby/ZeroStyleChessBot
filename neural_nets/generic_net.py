import torch
from sympy.abc import epsilon
from torch import nn
from torch.optim import Adam
from torch.optim import lr_scheduler
import torch
from functools import wraps
import os

import chess_moves
import numpy as np

chess_engine = chess_moves.ChessEngine()

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

    def __init__(self, input_size: tuple, output_size: tuple, init_lr: float = 0.001, save_dir: str = 'networks'):

        # Initialise nn Module
        super().__init__()

        # Control inputs and outputs
        self.input_size = input_size
        self.output_size = output_size

        self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        # Initialise the network
        self._build_network()

        self.optimiser = Adam(self.parameters(), lr=init_lr, weight_decay=1e-5)
        # Initialize the scheduler
        self.scheduler = lr_scheduler.StepLR(self.optimiser, step_size=1000, gamma=0.98)

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

    def create_legal_move_mask(self, moves, team: str = None):
        move_to_indices_lookup = []

        # Collect indices for batch updates
        all_indices = []

        for legal_move in moves:
            # Ignore pawn promotions for now
            if len(str(legal_move)) < 5:
                if team == "black":
                    legal_move_str = chess_engine.unflip_move(str(legal_move))
                    indices = chess_engine.move_to_target_indices(legal_move_str)
                else:
                    indices = chess_engine.move_to_target_indices(str(legal_move))

                all_indices.append(indices)
                move_to_indices_lookup.append([legal_move, indices])

        # Create mask efficiently with batched updates
        legal_move_mask = torch.zeros(self.output_size, dtype=torch.float32)
        if all_indices:
            all_indices = torch.tensor(all_indices, dtype=torch.long).flatten()  # Ensure 1D tensor
            legal_move_mask.index_fill_(0, all_indices, 1)

        return legal_move_mask, move_to_indices_lookup

    @staticmethod
    def get_move_from_tensor(tensor):
        # Returns the argmax move from the tensor
        channel = np.argmax(tensor) // (8 * 8)
        flat_index = np.argmax(tensor) % (8 * 8)
        from_row = flat_index // 8
        from_col = flat_index % 8

        return chess_engine.indices_to_move(channel, from_row, from_col)

    @staticmethod
    def board_to_tensor(board):
        fen = board.fen()
        return chess_engine.fen_to_tensor(fen)

    def node_evaluation(self, node):

        self.eval()
        with torch.no_grad():
            legal_move_mask, index_map = self.create_legal_move_mask(node.moves, node.team)
            board_tensor = chess_engine.fen_to_tensor(node.state)
            value, policy = self(torch.tensor(board_tensor, dtype=torch.float32, device='cuda').unsqueeze(0),
                                      legal_move_mask)

        # Define the alpha parameter for Dirichlet distribution and the weighting factor
        alpha = 0.3  # You can adjust this value depending on your needs
        epsilon = 0.25  # Weight for blending original policy and noise

        # Assosciate probability with its move
        for move, index in index_map:
            move.P = policy[0][index]

        Ps = np.array([move.P.cpu().item() if torch.is_tensor(move.P) else move.P for move in node.moves])

        sum_Ps = np.sum(Ps)

        if sum_Ps <= 1e-6:
            Ps = np.ones_like(Ps) / len(Ps)
            sum_Ps = np.sum(Ps)

        # Generate Dirichlet noise
        dirichlet_noise = np.random.dirichlet([alpha] * len(Ps))

        Ps = (1-epsilon) * Ps / sum_Ps + epsilon * dirichlet_noise

        for idx, move in enumerate(node.moves):
            move.P  = Ps[idx]

        if sum([move.P.cpu().item() if torch.is_tensor(move.P) else move.P for move in node.moves]) < 0.95:
            print(f"In tree: {sum([move.P.cpu().item() if torch.is_tensor(move.P) else move.P for move in node.moves])}")

        # Set flag saying it has been processed
        node.awaiting_processing = False

        # Return value for backpropagation
        return value.cpu().item()

    def tensorise_batch(self, states, moves, probabilities, wins):
        state_tens, legal_move_mask, _ = self.tensorise_inputs(states, moves)

        # Preallocate tensors
        batch_size = len(moves)
        moves_tens = torch.zeros((batch_size, *self.output_size), device=self.device)
        value_tens = torch.tensor(wins, dtype=torch.float32, device=self.device).unsqueeze(1)  # Vectorized

        # Flatten move sets for batch processing
        all_indices = []
        all_probs = []
        batch_indices = []

        for idx, (move_set, prob_set) in enumerate(zip(moves, probabilities)):
            for move, prob in zip(move_set, prob_set):
                indices = chess_engine.move_to_target_indices(str(move))
                all_indices.append(indices)  # Collect indices
                all_probs.append(prob)  # Collect probabilities
                batch_indices.append(idx)  # Keep track of batch position

        if all_indices:  # Avoid empty scatter calls
            all_indices = torch.tensor(all_indices, dtype=torch.long, device=self.device)
            all_probs = torch.tensor(all_probs, dtype=torch.float32, device=self.device)
            batch_indices = torch.tensor(batch_indices, dtype=torch.long, device=self.device)

            # Efficiently scatter values into moves_tens
            moves_tens[batch_indices, all_indices[:, 0], all_indices[:, 1], all_indices[:, 2]] = all_probs

        return state_tens, moves_tens, value_tens, legal_move_mask

    def tensorise_inputs(self, states, legal_moves):

        batch_size = len(states)

        # Preallocate tensors on GPU
        state_tens = torch.zeros((batch_size, *self.input_size), device=self.device)
        legal_move_mask = torch.zeros((batch_size, *self.output_size), device=self.device)

        # Collect move masks in CPU lists before batch transfer
        fen_tensors = []
        move_masks = []
        legal_move_keys = []

        for state, move_set in zip(states, legal_moves):
            # Convert FEN state to tensor (keep on CPU)
            fen_tensors.append(chess_engine.fen_to_tensor(state))

            # Determine player turn
            player_turn = "white" if state.split()[1] == 'w' else "black"

            # Generate legal move mask
            this_move_mask, key = self.create_legal_move_mask(move_set, player_turn)
            move_masks.append(this_move_mask)
            legal_move_keys.append(key)

        # Convert all tensors to GPU in one batch
        state_tens[:] = torch.from_numpy(np.array(fen_tensors)).to(self.device)
        legal_move_mask[:] = torch.stack(move_masks).to(self.device)  # Move entire batch at once

        return state_tens, legal_move_mask, legal_move_keys

class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, num_repeats: int = 2, activation_period: int = 2):

        super().__init__()

        convs_list = []

        for idx in range(num_repeats):
            convs_list.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            convs_list.append(nn.BatchNorm2d(in_channels))
            if idx % activation_period == 0:
                convs_list.append(nn.LeakyReLU())

        self.conv_block = nn.Sequential(*convs_list)
        self.out_act = nn.LeakyReLU()

    def forward(self, x: torch.tensor):

        # Perform convolutions
        out = self.conv_block(x)

        # Skip connection
        out = out + x
        out = self.out_act(out)

        return out


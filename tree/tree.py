import chess
import numpy as np
from copy import copy

from typing import List

class Tree:
    def __init__(self, board: chess.Board):
        # Store an instance of the board we ca play with
        self.board = board

        # Create the root node of the tree
        self.root = Node(board)
        self.nodes: List[Node] = []

    def search_down_tree(self, num_expands: int):

        for idx in range(num_expands):
            self.expand_the_tree(self.root)


    def expand_the_tree(self, current_node):

        # Create bool for while loop
        leaf = False

        # Create list of visited edges
        visited_edges: List[Edge] = []

        while not leaf:

            # Select the best child node according to PUCT
            edge = current_node.puct()

            # Add current node to the list of visited nodes
            visited_edges.append(edge)

            if edge.child_node is None:
                # If the node has no children, it's a leaf
                leaf = True
                # Set board to correct state with the nodes FEN
                self.board.set_fen(current_node.state)
                current_node, reward, done = edge.expand(self.board)
                self.nodes.append(current_node)
            else:
                # Else carry on searching
                current_node = edge.child_node

        self.backpropagation(visited_edges, reward)


    def backpropagation(self, visited_edges, observed_reward: float):

        gamma_factor = 0.99

        for edge in reversed(visited_edges):
            edge.N += 1
            edge.W += gamma_factor * observed_reward
            gamma_factor *= 0.99

class Node:
    def __init__(self, board, parent_edge=None):

        self.state= board.fen()  # The chess fen string object representing this state
        self.parent_edge = parent_edge  # Reference to the parent edge

        legal_moves = board.legal_moves

        self.number_legal_moves = len(list(legal_moves))

        self.edges= []  # List of child nodes (if any)
        for move in legal_moves:
            self.edges.append(Edge(self, move))

    def puct(self):

        Qs = np.array([edge.Q for edge in self.edges])
        Ns = np.array([edge.N for edge in self.edges])

        # PUCT calculation
        puct_vals = Qs + 5.0 * np.sqrt(np.sum(Ns)) / (1 + Ns)

        # Choose the child node with the highest PUCT value
        max_indices = np.where(puct_vals== np.max(puct_vals))[0]

        # Randomly choose one of the indices
        random_max_index = np.random.choice(max_indices)

        return self.edges[random_max_index]


class Edge:

    def __init__(self, parent_node, move):
        # Corresponding move
        self.move = move

        # Tree connections
        self.parent_node = parent_node
        self.child_node = None

        # Game statistics
        self.W = 0.
        self.N = 0.

    def expand(self, board: chess.Board):

        # Check if the move is a capture (not a pawn move)
        capture = board.is_capture(self.move)

        # Run the sim to update the board
        board.push(self.move)

        # Create a new child node
        self.child_node = Node(board, parent_edge=self)

        # Check if done
        done = board.is_game_over()

        # Calculate the reward
        if done:
            reward = 1. if board.result() == "1-0" else -1. if board.result() == "0-1" else 0.
        else:
            reward = 0.

        if capture:
            reward += 1.

        # Update statistics

        return self.child_node, reward, done

    @property
    def Q(self):
        return self.W / self.N if self.N > 0 else 0.



# Create the root of the tree
initial_board = chess.Board()
#
tree = Tree(initial_board)

tree.search_down_tree(30000)
print([edge.Q for edge in tree.root.edges])

best_moves = sorted(tree.root.edges, key= lambda x: x.Q, reverse=True)

for move in best_moves:
    initial_board.reset_board()
    initial_board.push(move.move)
    print(initial_board, "\n")

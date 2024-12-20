import numpy as np

class Node:

    def __init__(self, state, parent=None, move=None):

        self.parent = parent
        self.move = move
        self.child_edges = []

        # Save the state
        self.state = state
"""Read-only snapshots of a search tree, for showing what the engine is thinking. No UI code."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from zero_style_chess_engine.mcts import Node


@dataclass(frozen=True)
class CandidateMove:
    """One root move. Values are from the point of view of the side to move at the root."""

    move: str
    visits: int
    visit_share: float
    q: float
    prior: float


@dataclass(frozen=True)
class SearchSummary:
    """Immutable, so a search thread can hand it to a UI thread without locking."""

    fen: str
    total_visits: int  # Root visits, including those reused from the previous move's search
    rollouts_this_search: int
    elapsed_seconds: float
    candidates: List[CandidateMove]  # Most visited first
    principal_variation: List[str]  # Most visited line from the root
    net_value: Optional[float]  # The network's value of the root, before any search
    search_value: Optional[float]  # Best Q among visited root moves
    tree_size: int
    max_depth: int

    @property
    def white_to_move(self) -> bool:
        return self.fen.split()[1] == "w"

    @property
    def rollouts_per_second(self) -> float:
        return self.rollouts_this_search / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0


def summarise_search(root: Node, rollouts_this_search: int, elapsed_seconds: float, top_k: int = 5) -> SearchSummary:
    total_visits = int(root.Ns.sum())
    order = np.argsort(-root.Ns, kind="stable")[:top_k]
    candidates = [
        CandidateMove(
            move=root.moves[idx],
            visits=int(root.Ns[idx]),
            visit_share=float(root.Ns[idx] / total_visits) if total_visits else 0.0,
            q=float(root.Qs[idx]),
            prior=float(root.Ps[idx]),
        )
        for idx in order
        if root.Ns[idx] > 0
    ]
    visited = root.Ns > 0
    tree_size, max_depth = _tree_size_and_depth(root)

    return SearchSummary(
        fen=root.state,
        total_visits=total_visits,
        rollouts_this_search=rollouts_this_search,
        elapsed_seconds=elapsed_seconds,
        candidates=candidates,
        principal_variation=_principal_variation(root),
        net_value=root.net_value,
        search_value=float(root.Qs[visited].max()) if visited.any() else None,
        tree_size=tree_size,
        max_depth=max_depth,
    )


def _principal_variation(root: Node) -> List[str]:
    line: List[str] = []
    node = root
    while node.child_nodes:
        # argmax breaks visit ties by move index, as the candidate list does
        idx = int(np.argmax(node.Ns))
        line.append(node.moves[idx])
        node = node.child_nodes[idx]
    return line


def _tree_size_and_depth(root: Node) -> Tuple[int, int]:
    size, max_depth = 0, 0
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        size += 1
        max_depth = max(max_depth, depth)
        stack.extend((child, depth + 1) for child in node.child_nodes.values())
    return size, max_depth

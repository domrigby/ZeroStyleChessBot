import os
import warnings
from typing import Dict, List, Optional, Tuple

import chess_moves
import numpy as np
import pandas as pd

CURRICULUM_TAGS = ["mateIn1", "mateIn2", "mateIn3", "mateIn4", "short", "middleGame", "crushing", "mateIn5"]
START_STATE = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class LichessCuriculum:
    def __init__(self, game_states_dict: Dict[str, Tuple[List[str], List[str]]]):
        """
        :param game_states_dict: tag -> (puzzle FENs, puzzle move sequences)
        """
        self.game_states_dict = game_states_dict

    def get_start_state(self, episode: int) -> Tuple[str, int]:
        board = chess_moves.ChessEngine()
        if episode < 200:
            tag = "mateIn2"
        elif episode < 300:
            tag = "mateIn3"
        elif episode < 400:
            tag = "mateIn4"
        elif episode < 500:
            tag = "short"
        elif episode < 600:
            tag = "middleGame"
        elif episode < 700:
            tag = "mateIn5"
        elif episode < 800:
            tag = "crushing"
        else:
            return START_STATE, 300

        fens, moves = self.game_states_dict[tag]
        if not fens:
            return START_STATE, 300

        index = np.random.randint(len(fens))
        move = moves[index].split()[0]

        return board.push(fens[index], move), 100


def load_curricula(
    lichess_csv_db: Optional[str], num_agents: int, positions_per_tag: int = 1000
) -> List[Optional[LichessCuriculum]]:
    """
    Read the puzzle CSV once and give each agent its own random sample of positions per curriculum tag.
    :return: one curriculum per agent, or one None per agent (start-position games) if the CSV isn't available
    """
    if lichess_csv_db is None or not os.path.isfile(lichess_csv_db):
        if lichess_csv_db is not None:
            warnings.warn(f"Curriculum CSV {lichess_csv_db} not found: games start from the standard position")
        return [None] * num_agents

    df = pd.read_csv(lichess_csv_db, usecols=["FEN", "Moves", "Themes"])
    puzzles_by_tag = {tag: df[df["Themes"].str.contains(tag, case=False, na=False)] for tag in CURRICULUM_TAGS}

    curricula: List[Optional[LichessCuriculum]] = []
    for _ in range(num_agents):
        game_states_dict = {}
        for tag, puzzles in puzzles_by_tag.items():
            sample = puzzles.sample(n=min(positions_per_tag, len(puzzles)))
            game_states_dict[tag] = (sample["FEN"].tolist(), sample["Moves"].tolist())
        curricula.append(LichessCuriculum(game_states_dict))
    return curricula

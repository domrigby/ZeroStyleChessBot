import pandas as pd
import numpy as np
import subprocess
import chess_moves

class LichessCuriculum:

    def __init__(self, lichess_csv_db: str):

        # Load your CSV file into a DataFrame
        self.lichess_csv_db = lichess_csv_db
        self.sample_file = "sample.csv"
        self.sample_size = 2000000

        self.shuffle_command = f"(head -n 1 {self.lichess_csv_db} && tail -n +2 {self.lichess_csv_db} | shuf -n {self.sample_size}) > {self.sample_file}"

        # Shuffle the massive CSV and get a smaller sample
        subprocess.run(self.shuffle_command, shell=True, check=True)

        # Load the smaller sample into a DataFrame
        df = pd.read_csv(self.sample_file)

        curriculum = ['mateIn1', 'mateIn2', 'mateIn3', 'mateIn4', 'short', 'middleGame', 'crushing', 'mateIn5']

        self.game_states_dict = {}
        for state in curriculum:
            self.game_states_dict[state] = self.get_games_with_tag(df, state)

    def get_games_with_tag(self, df: pd.DataFrame, tag: str):
        # Filter rows where the Themes column contains the target tag
        filtered = df[df['Themes'].str.contains(tag, case=False, na=False)]
        return [filtered['FEN'].tolist(), filtered['Moves'].tolist()]

    def get_start_state(self, episode: int):
        board = chess_moves.ChessEngine()
        # if episode < 100:
        #     tag = 'mateIn1'
        if episode < 200:
            tag = 'mateIn2'
        elif episode < 300:
            tag = 'mateIn3'
        elif episode < 400:
            tag = 'mateIn4'
        elif episode < 500:
            tag = 'short'
        elif episode < 600:
            tag = 'middleGame'
        elif episode < 700:
            tag = 'mateIn5'
        elif episode < 800:
            tag = 'crushing'
        else:
            return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 300

        index = np.random.randint(len(self.game_states_dict[tag]))
        fen = self.game_states_dict[tag][0][index]
        move = self.game_states_dict[tag][1][index].split()[0]

        return board.push(fen, move), 100





if __name__ == '__main__':
    lcs = LichessCuriculum("/home/dom/Code/chess_bot/neural_nets/data/fen_chess_puzzles/lichess_db_puzzle_sorted.csv")
    lcs.get_start_state(50)
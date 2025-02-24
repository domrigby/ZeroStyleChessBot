import pandas as pd
import numpy as np
import subprocess

class LichessCuriculum:

    def __init__(self, lichess_csv_db: str):

        # Load your CSV file into a DataFrame
        self.lichess_csv_db = lichess_csv_db
        self.sample_file = "sample.csv"
        self.sample_size = 50000

        self.shuffle_command = f"(head -n 1 {self.lichess_csv_db} && tail -n +2 {self.lichess_csv_db} | shuf -n {self.sample_size}) > {self.sample_file}"

        # Shuffle the massive CSV and get a smaller sample
        subprocess.run(self.shuffle_command, shell=True, check=True)

        # Load the smaller sample into a DataFrame
        df = pd.read_csv(self.sample_file)

        curriculum = ['mateIn1', 'short', 'mateIn2', 'mateIn3', 'mateIn4', 'short', 'middleGame', 'crushing', 'mateIn5']

        self.game_states_dict = {}
        for state in curriculum:
            self.game_states_dict[state] = self.get_games_with_tag(df, state)


    def get_games_with_tag(self, df: pd.DataFrame, tag: str):
        # Filter rows where the Themes column contains the target tag
        filtered = df[df['Themes'].str.contains(tag, case=False, na=False)]
        return filtered['FEN'].tolist()

    def get_start_state(self, episode: int):
        if episode < 1000:
            return np.random.choice(self.game_states_dict['mateIn1'])
        if episode < 2000:
            return np.random.choice(self.game_states_dict['mateIn2'])
        if episode < 3000:
            return np.random.choice(self.game_states_dict['mateIn3'])
        if episode < 4000:
            return np.random.choice(self.game_states_dict['mateIn4'])
        if episode < 5000:
            return np.random.choice(self.game_states_dict['short'])
        if episode < 6000:
            return np.random.choice(self.game_states_dict['middleGame'])
        if episode < 7000:
            return np.random.choice(self.game_states_dict['mateIn5'])
        if episode < 8000:
            return np.random.choice(self.game_states_dict['crushing'])
        else:
            return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"




if __name__ == '__main__':
    lcs = LichessCuriculum("/home/dom/Code/chess_bot/neural_nets/data/fen_chess_puzzles/lichess_db_puzzle_sorted.csv")
    print('here')
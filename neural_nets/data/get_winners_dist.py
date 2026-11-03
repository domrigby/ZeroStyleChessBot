import argparse
import pickle
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot the distribution of winners across pickled moves.")
    parser.add_argument("moves_dir", help="Directory of pickled moves.")
    args = parser.parse_args()

    moves_dir = args.moves_dir

    moves = glob.glob(moves_dir + "/*")

    winners = np.zeros(len(moves))

    for idx, move in enumerate(moves):

        with open(move, 'rb') as f:
            data = pickle.load(f)

        winner = data["winner"]

        if winner == "white":
            winners[idx] = 1
        elif winner == "black":
            winners[idx] = -1
        else:
            winners[idx] = 0

    # Count occurrences of each unique value
    values, counts = np.unique(winners, return_counts=True)

    # Plot the histogram as a bar chart
    plt.bar(values, counts, tick_label=values, color=['red', 'blue', 'green'])

    # Label axes and title
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title('Histogram of -1, 0, 1')

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()

import argparse
import pickle
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot a histogram of values across pickled moves.")
    parser.add_argument("moves_dir", help="Directory of pickled moves.")
    args = parser.parse_args()

    moves_dir = args.moves_dir

    moves = glob.glob(moves_dir + "/*")

    values = np.zeros(len(moves))

    for idx, move in enumerate(moves):

        with open(move, 'rb') as f:
            data = pickle.load(f)

        print(data)

        value = data["value"]
        values[idx] = value

    # Create histogram
    plt.hist(values, bins=30, color='blue', edgecolor='black', alpha=0.7)

    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Data')

    # Show plot
    plt.show()


if __name__ == "__main__":
    main()

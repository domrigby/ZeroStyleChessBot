import argparse
import pickle
import numpy as np
import glob
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot a histogram of win values in an experience pickle.")
    parser.add_argument("data_path", help="Path to the experience .pkl file.")
    args = parser.parse_args()

    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    values = np.zeros(len(data))
    for idx, data_point in enumerate(data):
        values[idx] = data_point.win_val

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

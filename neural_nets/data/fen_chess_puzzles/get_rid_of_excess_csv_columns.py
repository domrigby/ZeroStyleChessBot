import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Keep only the FEN, Moves, Themes and Rating columns of the Lichess puzzle CSV.")
    parser.add_argument("input_csv", help="Path to the raw Lichess puzzle CSV.")
    parser.add_argument("output_csv", help="Path to write the reduced CSV.")
    args = parser.parse_args()

    # Define file path
    input_csv = args.input_csv
    output_csv = args.output_csv

    # Load only selected columns to save memory
    columns_to_keep = ['FEN', 'Moves', 'Themes', 'Rating']
    df = pd.read_csv(input_csv, usecols=columns_to_keep)

    # Save the reduced dataset to a new CSV
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()

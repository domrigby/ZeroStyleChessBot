import argparse
import os
import pickle
import random

MAX_SIZE = 250_000  # Max entries per file


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_pickle(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def main():
    parser = argparse.ArgumentParser(description="Shuffle test_moves_*.pkl files into new chunks.")
    parser.add_argument("source_dir", help="Directory containing the existing test_moves_*.pkl files.")
    parser.add_argument("dest_dir", help="Directory to write the shuffled files to.")
    args = parser.parse_args()

    SOURCE_DIR = args.source_dir
    DEST_DIR = args.dest_dir

    os.makedirs(DEST_DIR, exist_ok=True)

    buffer = []
    file_counter = 0

    input_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.startswith("test_moves_") and f.endswith(".pkl")])

    for fname in input_files:
        print(f"Reading file: {fname}")
        data = load_pickle(os.path.join(SOURCE_DIR, fname))
        buffer.extend(data)

        # Shuffle when buffer size gets large
        if len(buffer) > MAX_SIZE * 10:
            random.shuffle(buffer)

        # Write full chunks
        while len(buffer) >= MAX_SIZE * 10:
            chunk, buffer = buffer[:MAX_SIZE], buffer[MAX_SIZE:]
            out_fname = f"test_moves_{file_counter}.pkl"
            save_pickle(chunk, os.path.join(DEST_DIR, out_fname))
            print(f"Saved {out_fname} with {len(chunk)} items.")
            file_counter += 1

    # Write any remaining data
    if buffer:
        random.shuffle(buffer)
        out_fname = f"test_moves_{file_counter}.pkl"
        save_pickle(buffer, os.path.join(DEST_DIR, out_fname))
        print(f"Saved {out_fname} with {len(buffer)} items.")


if __name__ == "__main__":
    main()

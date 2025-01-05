import os
import pickle
import random

# Configuration
input_directory = "/home/dom/Code/chess_bot/neural_nets/data/pickles"  # Directory containing original pickle files
output_directory = "shuffled_chunks"  # Directory to save shuffled files
chunk_size = 100000  # Number of moves per output file
shuffle_buffer_size = 500000  # Size of the shuffle buffer (adjust for available memory)

os.makedirs(output_directory, exist_ok=True)

# Step 1: Load all moves into a buffer (in chunks)
buffer = []
chunk_count = 0

# Read all pickle files
for filename in os.listdir(input_directory):
    filepath = os.path.join(input_directory, filename)

    if not filename.endswith('.pkl'):  # Skip non-pickle files
        continue

    with open(filepath, 'rb') as f:
        while True:
            try:
                moves = pickle.load(f)
                buffer.extend(moves)

                # When buffer exceeds shuffle_buffer_size, shuffle and process it
                if len(buffer) >= shuffle_buffer_size:
                    random.shuffle(buffer)  # Shuffle the buffer

                    # Write out chunks from the shuffled buffer
                    while len(buffer) >= chunk_size:
                        output_filename = os.path.join(output_directory, f"shuffled_chunk_{chunk_count}.pkl")
                        with open(output_filename, 'wb') as out_f:
                            pickle.dump(buffer[:chunk_size], out_f)
                        buffer = buffer[chunk_size:]
                        chunk_count += 1

            except EOFError:
                break

# Step 2: Final shuffle and write remaining moves
if buffer:
    random.shuffle(buffer)
    while len(buffer) > 0:
        output_filename = os.path.join(output_directory, f"shuffled_chunk_{chunk_count}.pkl")
        with open(output_filename, 'wb') as out_f:
            pickle.dump(buffer[:chunk_size], out_f)
        buffer = buffer[chunk_size:]
        chunk_count += 1

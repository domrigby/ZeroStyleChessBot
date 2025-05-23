import os
import pickle

# Configuration
input_dir = "/home/dom/1TB_drive/chess_data"   # Change this to your actual path
max_items = 250000

# Process each pickle file
for filename in os.listdir(input_dir):
    if filename.startswith("train_moves_") and filename.endswith(".pkl"):
        file_path = os.path.join(input_dir, filename)
        print(f"Processing {file_path}...")

        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            if isinstance(data, list) and len(data) > max_items:
                data = data[:max_items]
                with open(file_path, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Trimmed and saved: {file_path}")
            else:
                print(f"No trimming needed: {file_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

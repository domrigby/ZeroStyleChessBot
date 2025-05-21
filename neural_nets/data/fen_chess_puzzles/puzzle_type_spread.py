import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Load CSV
df = pd.read_csv('/home/dom/Code/chess_bot/neural_nets/data/fen_chess_puzzles/lichess_db_puzzle_sorted.csv')  # Replace with your actual filename

# Split themes by row and flatten
theme_lists = df['Themes'].dropna().str.split()
all_themes = [theme for sublist in theme_lists for theme in sublist]

# Count occurrences
theme_counts = Counter(all_themes)

# Convert to DataFrame for plotting
theme_df = pd.DataFrame(theme_counts.items(), columns=['Theme', 'Count']).sort_values('Count', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.bar(theme_df['Theme'], theme_df['Count'])
plt.xticks(rotation=45, ha='right')
plt.ylabel('Number of Positions')
plt.title('Distribution of Themes in Dataset')
plt.tight_layout()
plt.show()

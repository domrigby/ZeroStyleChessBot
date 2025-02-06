import pickle
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

moves_dir = r"/home/dom/Code/chess_bot/neural_nets/data/real_moves/76"

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
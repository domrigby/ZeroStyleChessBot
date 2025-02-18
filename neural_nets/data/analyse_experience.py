import pickle
import numpy as np
import glob
import matplotlib.pyplot as plt

with open("/home/dom/Code/chess_bot/new_data.pkl", 'rb') as f:
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
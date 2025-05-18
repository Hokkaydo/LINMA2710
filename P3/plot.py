import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

naive = pd.read_csv('results/naive.csv')
fast = pd.read_csv('results/fast.csv')

speedup = naive['Time'] / fast['Time']

plt.figure(figsize=(10, 6))
plt.semilogx(naive['Size'], speedup, marker='o', linestyle='-', color='b', label='Speedup')
plt.title('Speedup of Fast Algorithm over Naive Algorithm')
plt.xlabel('Input Size')
plt.ylabel('Speedup Factor')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('results/speedup_plot.png')
plt.show()

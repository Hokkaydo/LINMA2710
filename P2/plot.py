import pandas as pd
import matplotlib.pyplot as plt

for i in range(0, 6):
    i = 2**i
    df = pd.read_csv(f"results/benchmark_results_{i}_procs.txt")

    plt.plot(df['Size'], df['Time'], label=f"{i} procs", marker='o', linewidth=3, markersize=8)

plt.xlabel('Size')
plt.ylabel('Time (s)')
plt.title('Benchmark Results')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.savefig('results/benchmark_results.png')
plt.show()
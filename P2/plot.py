import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

val = None

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

val_16procs = pd.read_csv("results/benchmark_results_16_procs.csv")

for i in range(0, 6):
    i = 2**i
    df = pd.read_csv(f"results/benchmark_results_{i}_procs.csv")
    
    if i == 2:
        val = df['Time']
    if i == 16:
        val /= df['Time']
    axes[0].plot(df['Size'], df['Time'], label=f"{i} procs", marker='o', linewidth=2, markersize=6)
    axes[1].plot(df['Size'], df['Time'] / val_16procs['Time'][:len(df['Time'])], label=f"{i} procs", marker='o', linewidth=2, markersize=6)

axes[0].set_xlabel('Size')
axes[0].set_ylabel(r'Time ($\mu$s)')
axes[0].set_title('Benchmarks of "DistributedMatrix::multiplyTransposed"\n operation for multiple processors and sizes')
axes[0].legend()
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].minorticks_on()
axes[0].grid(which='both', color='lightgrey', linestyle='-', linewidth=0.5)

axes[1].set_xlabel('Size')
axes[1].set_ylabel('Speedup')
axes[1].set_title('Speedup of "DistributedMatrix::multiplyTransposed" operation\n between 16 processors and other processor amounts')
axes[1].legend()
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].minorticks_on()
axes[1].grid(which='both', color='lightgrey', linestyle='-', linewidth=0.5)



fig.savefig('results/benchmark.pdf')
plt.show()
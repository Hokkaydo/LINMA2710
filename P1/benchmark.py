import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import perf_counter as clock
import sys

results = [
    ("bench-results_save.csv", "Naive", "r", 1),
    ("bench-results-speed.csv", "AVX2, FMA, OMP", "g", 3),
    ("bench-results-speed-arch-native.csv", "AVX2, FMA, OMP, -march=native", "b", 2),
]

if len(sys.argv) > 1 and sys.argv[1] == "matmul":
    import subprocess
    start = clock()
    subprocess.run(["./bench-runner"])
    print(f"Matmul took {(clock() - start)/1000} seconds for 1000x1000 * 1000x1000")
    sys.exit()

for filename, title, color, width in results:
    data = pd.read_csv(filename)
    x = data['m']*data['n']*data['k']
    y = data['time']
    # average y's for same x's
    y = y.groupby(x).mean()
    x = np.array(y.index)
    y = y.values
    coeff = np.polyfit(np.log(x), np.log(y), 1)
    poly = np.poly1d(coeff)
    fit = np.exp(poly(np.log(x)))
    plt.loglog(x, fit, color + '--', label=title + ' fit', linewidth=width)
    plt.scatter(x, y, color=color, label=title, s=10, linewidths=width)
    
t = np.linspace(0, data['m'].max()*data['n'].max()*data['k'].max(), 100)


plt.xlabel(r'$m\times n\times k$ $[-]$')
plt.ylabel(r'Time $[s]$')
plt.title(r'Matrix multiplication time for 2 matrices of size $m\times n$ and $n\times k$ for variable $m,n,k$', wrap=True)
plt.legend()
plt.savefig("benchmark.pdf")
plt.show()

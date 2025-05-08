import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import perf_counter as clock
import sys

import matplotlib as mpl
# use LaTeX fonts in the plot
mpl.rc('text', usetex=True)
mpl.rc('font', family='serif')


results = [
    ("bench-results_save.csv", "Naive", "r", 2),
    # ("bench-results-speed.csv", "AVX2, FMA, OMP", "g", 2),
    # ("bench-results-speed-arch-native.csv", "AVX2, FMA, -mtune=native", "b", 2),
    ("bench-results-fast-final.csv", "AVX2, FMA, -mtune=native, blocked tile", "b", 2),
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
    plt.scatter(x, y, color=color, label=title, s=10, linewidths=width)
    # plt.loglog(x, fit, color + '--', label=title + ' fit', linewidth=2)
    
plt.loglog(x, 0.8*10e-11*x, 'k--', label=r'$\mathcal{O}(m\times n\times k)$', linewidth=2)
plt.loglog(x, 3.9*10e-10*x, 'k--', linewidth=2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$m\times n\times k$ [-]')
plt.ylabel(r'Time [s]')
plt.ylim(10e-6, 10)
plt.title('Matrix multiplication time for 2 matrices of size $m\\times n$ and $n\\times k$ \n for variable $m,n,k$')
plt.legend(loc='upper left')
plt.grid()
plt.savefig("benchmark.pdf")
# plt.show()

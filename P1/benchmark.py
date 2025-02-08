import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

if len(sys.argv) != 2:
    print('Usage: python3 benchmark.py <data.csv>')
    sys.exit(1)
data = pd.read_csv(sys.argv[1])

plt.scatter(data['m']*data['n']*data['k'], data['time']/1000, label='Time', color='red', s=10)
t = np.linspace(0, data['m'].max()*data['n'].max()*data['k'].max(), 100)
plt.plot(t, 4.5*10**(-9)*t, label=r'$\mathcal{O}(m\times n\times k)$', color='blue')
plt.xlabel(r'$m\times n\times k$ $[-]$')
plt.ylabel(r'Time $[s]$')
plt.title(r'Matrix multiplication time for 2 matrices of size $m\times n$ and $n\times k$ for variable $m,n,k$', wrap=True)
plt.legend()
plt.savefig('benchmark.pdf')
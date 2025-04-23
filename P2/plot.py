import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set_theme()
import numpy as np
import subprocess
import re

def plot_benchmark():
    val = None

    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    val_16procs = pd.read_csv("results/benchmark_results_16_procs.csv")[1:]

    for i in range(0, 6):
        i = 2**i
        df = pd.read_csv(f"results/benchmark_results_{i}_procs.csv")[1:]
        
        if i == 2:
            val = df['Time']
        if i == 16:
            val /= df['Time']
        axes[0].plot(df['Size'], df['Time'], label=f"{i} procs", marker='o', linewidth=2, markersize=6)
        axes[1].plot(df['Size'], df['Time'] / val_16procs['Time'][:len(df['Time'])], label=f"{i} procs", marker='o', linewidth=2, markersize=6)

    x = np.linspace(df['Size'].min(), df['Size'].max(), 100)

    axes[0].plot(x, x**2.5, label=r'$\mathcal{O}(n^{2.5})$', linestyle='--', color='black')
    axes[0].set_xlabel('Size')
    axes[0].set_ylabel('Time (ns)')
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
    
def parse_nvtx_sum(filename):
    command = [
        'nsys', 'stats', '--report', 'nvtx_sum', filename
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    nsys_output = result.stdout

    lines = nsys_output.strip().split('\n')
    data = []

    for i, line in enumerate(lines):
        if re.match(r"\s*-+\s*-+", line):
            data_lines = lines[i+1:] 
            break
    else:
        data_lines = [] 

    for line in data_lines:
        parts = re.split(r'\s{2,}', line.strip())
        if len(parts) >= 10:
            time_percent = float(parts[0].replace(',', '.'))
            instances = int(parts[2])
            range_desc = parts[9]
            data.append((time_percent, instances, range_desc))

    return pd.DataFrame(data, columns=['Time (%)', 'Instances', 'Range'])

def parse_osrt_sum(filename):
    command = [
        'nsys', 'stats', '--report', 'osrt_sum', filename
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    nsys_output = result.stdout

    lines = nsys_output.strip().split('\n')
    data = []

    for i, line in enumerate(lines):
        if re.match(r"\s*-+\s*-+", line):
            data_lines = lines[i+1:] 
            break
    else:
        data_lines = [] 

    for line in data_lines:
        parts = re.split(r'\s{2,}', line.strip())
        time_percent = float(parts[0].replace(',', '.'))
        name = parts[-1]
        data.append((time_percent, name))

    return pd.DataFrame(data, columns=['Time (%)', 'Name'])


def plot_nsys():
      
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    
    nvtx_time_lines = {}
    nvtx_instances_lines = {}
    osrt_lines = {}
    
    nvtx_colors = {}
    osrt_colors = {}
        
    for i in range(0, 6):
        i = 2**i

        nvtx_sum = parse_nvtx_sum(f"results/nsys/procs_{i}.sqlite")
        osrt_sum = parse_osrt_sum(f"results/nsys/procs_{i}.sqlite").sort_values(by='Time (%)', ascending=False)[:2]

        nvtx_cols = nvtx_sum['Range'].tolist()
        osrt_cols = osrt_sum['Name'].tolist()
        
        osrt_cols = osrt_sum['Name'].tolist()
        for j in range(len(osrt_sum)):
            if osrt_cols[j] not in osrt_lines:
                osrt_lines[osrt_cols[j]] = []
                osrt_colors[osrt_cols[j]] = colors[j]
            osrt_lines[osrt_cols[j]].append((i, osrt_sum['Time (%)'][j]))

        for j in range(len(nvtx_sum)):
            if nvtx_cols[j] not in nvtx_time_lines:
                nvtx_time_lines[nvtx_cols[j]] = []
                nvtx_instances_lines[nvtx_cols[j]] = []
                nvtx_colors[nvtx_cols[j]] = colors[j]
                
            nvtx_time_lines[nvtx_cols[j]].append([i, nvtx_sum['Time (%)'][j]])
            nvtx_instances_lines[nvtx_cols[j]].append([i, nvtx_sum['Instances'][j]])        
        
    
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

    # Big figure on the left, spans both rows
    ax_big = fig.add_subplot(gs[:, 0])  # ":" means both rows

    # Two small figures on the right, one per row
    ax_small1 = fig.add_subplot(gs[0, 1])
    ax_small2 = fig.add_subplot(gs[1, 1])
    
    for key, values in osrt_lines.items():
        ax_small1.plot(*np.array(values).T, label=key, marker='o', linewidth=2, markersize=6, color=osrt_colors[key])
    for key, values in nvtx_time_lines.items():
        ax_big.plot(*np.array(values).T, label=key, marker='o', linewidth=2, markersize=6, color=nvtx_colors[key])
        
    ax_small2.plot(*np.array(nvtx_instances_lines["MPI:MPI_Finalize"]).T, label="MPI:MPI_Finalize", marker='o', linewidth=2, markersize=10, color=nvtx_colors["MPI:MPI_Finalize"])
    for key, values in nvtx_instances_lines.items(): 
        if key == "MPI:MPI_Finalize":
            continue
        ax_small2.plot(*np.array(values).T, label=key, marker='o', linewidth=2, markersize=6, color=nvtx_colors[key])    
       
    x = np.linspace(1, 32, 100)
    ax_small2.plot(x, 1.5*x, label=r'$\mathcal{O}(n)$', linestyle='--', color='black')
 
    for ax in [ax_big, ax_small1, ax_small2]:
        ax.set_xlabel('Number of processors')
        ax.set_ylabel('Time spent (%)')
        ax.set_ylim(-1, 101)
        ax.legend()
        ax.minorticks_on()
        ax.grid(which='both', color='lightgrey', linestyle='-', linewidth=0.5)
    
    ax_big.set_title('Time spent on MPI collectives \n w.r.t. number of processors')
    ax_small1.set_title('Time spent on the most used operations \n w.r.t. number of processors')
    
    ax_small2.set_title('Number of MPI collectives calls \n w.r.t. number of processors')
    ax_small2.set_ylabel('Number of calls')
    ax_small2.set_ylim(auto=True)
    ax_small2.set_yscale('log')
    ax_small2.set_xscale('log')

    fig.tight_layout()
    fig.savefig('results/nsys.pdf')
    
    
        

plot_benchmark()
# plot_nsys()
plt.show()
    

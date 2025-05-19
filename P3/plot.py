import matplotlib.pyplot as plt
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Plot speedup of fast algorithm over naive algorithm.')
parser.add_argument('--naive', type=str, required=True, help='Path to naive algorithm results CSV file.')
parser.add_argument('--fast', type=str, required=True, help='Path to fast algorithm results CSV file.')
parser.add_argument("--emission", type=str, default="none", help="Emissions directory")
parser.add_argument("--output", type=str, default="results", help="Output directory for the plot")

args = parser.parse_args()

naive = pd.read_csv(args.naive)
fast = pd.read_csv(args.fast)

speedup = naive['Time'] / fast['Time']

plt.figure(figsize=(10, 6))
plt.semilogx(naive['Size'], speedup, marker='o', linestyle='-', color='b', label='Speedup')
plt.title('Speedup of Fast Algorithm over Naive Algorithm')
plt.xlabel('Input Size')
plt.ylabel('Speedup Factor')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(f'{args.output}/speedup_plot.png')
#plt.show()
plt.clf()

# list files in emissions directory
import os
if args.emission != "none":
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    
    axes_data_id = { "emissions": axes[0], "energy_consumed": axes[1] }
    
    for method in ["Naive", "Fast"]:
        emissions = {}

        for root, dirs, files in os.walk(f"{args.emission}/{method.lower()}"):
            for file in files:
                size = int(file.split("_")[1].split(".")[0])
                emissions[size] = pd.read_csv(os.path.join(root, file))
        # aggregate emissions in one dataframe adding the size as a column
        if len(emissions) > 0:
            emissions_df = pd.DataFrame.from_dict(emissions, orient='index')
            emissions_df['Size'] = emissions_df.index
            emissions_df = emissions_df.reset_index(drop=True)
        emissions_df['emissions'] = emissions_df['emissions'].astype(float)*1000                # to gCO2eq
        emissions_df['energy_consumed'] = emissions_df['energy_consumed'].astype(float)*1000    # to Wh
            
        for data_id, ax in axes_data_id.items():
            ax.semilogx(emissions_df['Size'], emissions_df[data_id], marker='o', linestyle='-', label=method)
            
    
    axes[0].set_title('Emissions of Matrix Multiplication Algorithms')
    axes[0].set_xlabel('Input Size')
    axes[0].set_ylabel('Emissions - [gCO2eq]')
    axes[0].legend()
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    
    axes[1].set_title('Energy Consumed of Matrix Multiplication Algorithms')
    axes[1].set_xlabel('Input Size')
    axes[1].set_ylabel('Energy Consumed - [Wh]')
    axes[1].legend()
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    
    fig.tight_layout()
    plt.savefig(f'{args.output}/emissions_plot.png')
    #plt.show()


#!/bin/bash
#SBATCH --job-name=pluto_gpu1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=10000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

ip=$(hostname -i)
echo "Run \$ sshuttle -r $CLUSTER_NAME $ip/16"
srun julia --project -e "import Pluto; Pluto.run(host=\"$ip\", port=1234, launch_browser=false)"

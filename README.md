# LINMA2710 Scientific Computing

[<img src="https://upload.wikimedia.org/wikipedia/commons/7/72/UCLouvain_logo.svg" height="20"/>](https://uclouvain.be/en-cours-2024-linma2710)
[<img src="https://upload.wikimedia.org/wikipedia/commons/c/c6/Moodle-logo.svg" height="16"/>](https://moodle.uclouvain.be/enrol/index.php?id=2951)

This repository contains different resources for the LINMA2710 courses given at UCLouvain.

## Schedule

| Week | Thursday   | Topic        | Lecturer |
|------|------------|--------------|----------|
| S1    | 06/02/2025 | C++          | Absil    |
| S2    | 13/02/2025 | C++          | Absil    |
| S3    | 20/02/2025 | Parallel     | Legat    |
| S4    | 27/02/2025 | Parallel     | Legat    |
| S5    | 06/03/2025 | Distributed  | Legat    |
| S6    | 13/03/2025 | Distributed  | Legat    |
| S7    | 20/03/2025 | GPU          | Legat    |
| S8    | 27/03/2025 | GPU          | Legat    |
| S9    | 03/04/2025 | PDE          | Absil    |
| S10   | 10/04/2025 | PDE          | Absil    |
| S11   | 17/04/2025 | PDE          | Absil    |
| 🥚    | 24/04/2025 | 🐇           | 🐰       |
| 🥚    | 01/05/2025 | 🐇           | 🐰       |
| S12   | 08/05/2025 | Project help |          |
| S13   | 15/05/2025 | Project help |          |

## CECI cluster

In order to use the CECI clusters, you need a CECI account.
If you don't already have an account (if you don't know whether you have an account, chances are you don't have one), first [create one](https://login.ceci-hpc.be/init/).
You will receive an email, follow the link in the email and in the field labelled "Email of Supervising Professor", enter `benoit.legat@uclouvain.be`.
Follow the steps detailed [here](https://support.ceci-hpc.be/doc/_contents/QuickStart/ConnectingToTheClusters/index.html) in order to download your private key, create the corresponding public key and create the file `.ssh/config`.

You should now be able to connect to the manneback cluster with
```sh
(your computer) $ ssh manneback
```

Follow [this guide](https://support.ceci-hpc.be/doc/_contents/ManagingFiles/TransferringFilesEffectively.html) to copy files from your computer to the cluster. For instance, with `scp` you can copy a file `submit.sh` from your computer with:
```sh
(your computer) $ scp submit.sh manneback:.
```
It might however be a bit tedious to keep the files in sync with `scp`. I recommend pushing your project in a **private** (don't use a public git as your code shouldn't be accessible to other students!) git (for instance in https://forge.uclouvain.be/) and pull it from the CECI cluster. You can then easily update the code on the CECI cluster with `git pull`.
**Important** do not sync the binaries of with the CECI cluster as you might have a different architecture. Exclude them from the git by adding them in the `.gitignore` file and simply recompile them on the cluster.

To run your code, [submit a job with Slurm](https://support.ceci-hpc.be/doc/_contents/QuickStart/SubmittingJobs/SlurmTutorial.html).
The file `examples/submit.sh` gives an example of submission script to use (see [here](https://www.ceci-hpc.be/scriptgen.html) for a helper for writing your own submission script). You can use it with
```sh
(manneback cluster) $ sbatch submit.sh
```ion 
The output produced by the job is written in the file `slurm-<JOBID>.out` where `<JOBID>` is the job id listed in the `JOBID` column of the table outputted by
```sh
(manneback cluster) $ squeue --me
```

Note that if you do `(manneback cluster) $ ./a.out` directly without using `sbatch` or `srun`, the notebook will run on the *login node* which has limited resources as it is only meant for you to connect and send jobs via Slurm that are executed on *compute nodes*, you will also not have any GPU on the login node.

If you use `srun` directly without using `sbatch`, the output will be displayed directly on the terminal and not to a `slurm-<JOBID>.out` file.
This means that you will loose the output if you loose the `ssh` connection (which can easily happen, e.g., if you laptop is suspended).
One very useful trick is to use `screen`. If your `ssh` connection is lost, simply reconnect and run `screen -r` to get your session back. More details [here](https://linuxize.com/post/how-to-use-linux-screen/).

## Julia

**Do not** use `module load CUDA`. This command uses [Lmod](https://github.com/TACC/Lmod) to set `LD_LIBRARY_PATH` (as detailed in the output of `module show CUDA`) [which is discouraged](https://github.com/JuliaGPU/CUDA.jl/issues/1755).

Running Julia interactively on a compute node is as simple as running `$ srun --pty julia`.
If `CUDA` was precompiled on a node with no GPU (such as the login node), you will see the error
```julia
julia> using CUDA
┌ Error: CUDA.jl could not find an appropriate CUDA runtime to use.
│ 
│ CUDA.jl's JLLs were precompiled without an NVIDIA driver present.
│ This can happen when installing CUDA.jl on an HPC log-in node,
│ or in a container. In that case, you need to specify which CUDA
│ version to use at run time by calling `CUDA.set_runtime_version!`
│ or provisioning the preference it sets at compile time.
│ 
│ If you are not running in a container or on an HPC log-in node,
│ try re-compiling the CUDA runtime JLL and re-loading CUDA.jl:
│      pkg = Base.PkgId(Base.UUID("76a88914-d11a-5bdc-97e0-2f5a05c973a2"),
│                       "CUDA_Runtime_jll")
│      Base.compilecache(pkg)
│      # re-start Julia and re-load CUDA.jl
│ 
│ For more details, refer to the CUDA.jl documentation at
│ https://cuda.juliagpu.org/stable/installation/overview/
└ @ CUDA ~/.julia/packages/CUDA/1kIOw/src/initialization.jl:118
```
Just copy-paste these lines on the REPL to re-compile CUDA.jl and then exit it and restart a new Julia session with `srun` again.
If you still get the error, leave the REPL, then run the following (replacing `v1.11` by your Julia version of course):
```sh
(manneback cluster) $ rm -r ~/.julia/compiled/v1.11/CUDA*
```
New, start a new Julia session with `srun` and `using CUDA` should not error anymore.

See [here](https://enccs.github.io/julia-for-hpc/hpc-cluster/) for additional information.

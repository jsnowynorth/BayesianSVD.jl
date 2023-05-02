#!/bin/bash
#--------------------------------------------------------------------------------
#  SBATCH CONFIG
#--------------------------------------------------------------------------------
#SBATCH -N 1                                    # number of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -C cpu                                  # constraint (haswell or knl for cori and cpu for perlmutter)
#SBATCH -q regular                              # quality of service (qos)
#SBATCH -J PDO                                  # name of job
#SBATCH --mail-user=jsnorth@lbl.gov             # email
#SBATCH --mail-type=ALL                         # when to email notification
#SBATCH -t 0-12:00:00                           # time in days-hours:minutes:seconds
#SBATCH -A m1517                                # project to charge for the job
#--------------------------------------------------------------------------------

#OpenMP settings:
export OMP_NUM_THREADS=4
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

echo "### Starting at: $(date) ###"

## Module Commands
ml load julia
module load cray-hdf5/1.12.2.1
module load cray-netcdf/4.9.0.1

## run the application
srun -n 1 -c 8 --cpu_bind=cores julia examples/PDO.jl
# srun -n 1 -c 8 --cpu_bind=cores julia examples/PDO.jl 2 # increment the number based on .jld2 file  


echo "### Ending at: $(date) ###"

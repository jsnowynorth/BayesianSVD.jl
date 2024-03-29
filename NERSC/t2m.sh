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
#SBATCH -t 0-10:00:00                           # time in days-hours:minutes:seconds
#SBATCH -A m1517                                # project to charge for the job
#--------------------------------------------------------------------------------

#OpenMP settings:
export OMP_NUM_THREADS=4
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

echo "### Starting at: $(date) ###"

## Module Commands
ml load julia
module load cray-hdf5
module load cray-netcdf

## ARG1: fileNumber = parse(Int64, ARGS[1]) # should be the next number in line (i.e., if run_2.jl exists, then should be 3)
## ARG2: samplingIndicator = parse(Bool, ARGS[2]) # if true, sample from model post-burnin. if false, burnin
## ARG3: nits = parse(Int64, ARGS[3]) # number of samples to draw
## ARG4: burnin = parse(Int64, ARGS[4]) # number of samples to burn (when equal to nits, no samples will be saved and still in burnin phase)

## run the application
srun -n 1 -c 8 --cpu_bind=cores julia t2mV2.jl # first run for 1000 and burn 1000 - initializes model
# srun -n 1 -c 8 --cpu_bind=cores julia t2mcontinue.jl 2 false 1000 1000 # BURNIN: file false=burnin numSamps numBurn
# srun -n 1 -c 8 --cpu_bind=cores julia t2mcontinue.jl 6 true 1000 0 # SAMPLE: file true=sample numSamps numBurn


echo "### Ending at: $(date) ###"

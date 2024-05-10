#!/bin/bash
#--------------------------------------------------------------------------------
#  SBATCH CONFIG
#--------------------------------------------------------------------------------
#SBATCH --nodes=20                                    # number of nodes
#SBATCH --ntasks-per-node=32
#SBATCH -C cpu                                  # constraint (haswell or knl for cori and cpu for perlmutter)
#SBATCH -q regular                              # quality of service (qos)
#SBATCH -J VariableLengthSimulation             # name of job
#SBATCH --mail-user=jsnorth@lbl.gov             # email
#SBATCH --mail-type=ALL                         # when to email notification
#SBATCH -t 01:00:00                             # time in days-hours:minutes:seconds
#SBATCH -A m1517                                # project to charge for the job
#--------------------------------------------------------------------------------

# nnode = n sims / (32 = task per node = 128/threads)

echo "### Starting at: $(date) ###"

## Module Commands
ml load julia

## export OMP_NUM_THREADS=4
## export OMP_PLACES=threads
## export OMP_PROC_BIND=spread


## run the application
julia VariableLengthSimulation.jl


echo "### Ending at: $(date) ###"

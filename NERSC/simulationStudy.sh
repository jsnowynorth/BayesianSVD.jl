#!/bin/bash
#--------------------------------------------------------------------------------
#  SBATCH CONFIG
#--------------------------------------------------------------------------------
#SBATCH --nodes=94                                    # number of nodes
#SBATCH --ntasks-per-node=32
#SBATCH -C cpu                                  # constraint (haswell or knl for cori and cpu for perlmutter)
#SBATCH -q debug                                # quality of service (qos)
#SBATCH -J simulationTest                       # name of job
#SBATCH --mail-user=jsnorth@lbl.gov             # email
#SBATCH --mail-type=ALL                         # when to email notification
#SBATCH -t 00:10:00                             # time in days-hours:minutes:seconds
#SBATCH -A m1517                                # project to charge for the job
#--------------------------------------------------------------------------------

# nnode = n sims / (32 = task per node = 128/threads)

echo "### Starting at: $(date) ###"

## Module Commands
# module load julia/1.8.0-beta1

# module load PrgEnv-cray
# module load julia/1.8.0-beta1-cray
ml load julia

export OMP_NUM_THREADS=4
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


## run the application
julia simulationStudy.jl


echo "### Ending at: $(date) ###"

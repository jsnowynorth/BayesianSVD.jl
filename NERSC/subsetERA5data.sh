!/bin/bash
#--------------------------------------------------------------------------------
#  SBATCH CONFIG
#--------------------------------------------------------------------------------
#SBATCH --nodes=1                                    # number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH -C cpu                                  # constraint (haswell or knl for cori and cpu for perlmutter)
#SBATCH -q regular                                # quality of service (qos)
#SBATCH -J selectLonLat                         # name of job
#SBATCH --mail-user=jsnorth@lbl.gov             # email
#SBATCH --mail-type=ALL                         # when to email notification
#SBATCH -t 06:00:00                             # time in days-hours:minutes:seconds
#SBATCH -A m1517                                # project to charge for the job
#--------------------------------------------------------------------------------


echo "### Starting at: $(date) ###"

## Module Commands
module spider cray-netcdf
module load e4s
spack env activate gcc
spack load cdo


## fileDir=/global/cfs/projectdirs/m3522/cmip6/ERA5/e5.mx2t_daily/

startYear=1979
endYear=2021
startMonth=01
endMonth=12


for years in $(seq -w ${startYear} ${endYear}); do
        for months in $(seq -w ${startMonth} ${endMonth}); do
                directory="/global/cfs/projectdirs/m3522/cmip6/ERA5/e5.mx2t_daily/e5.mx2t_daily.${years}${months}.nc"
                directoryNew="/pscratch/sd/j/jsnorth/ERA5MonthlyMax/Data/mx2t_daily_${years}${months}.nc"
                # cdo sellonlatbox,-128,-116,44,53 directory directoryNew # only subsets data
                cdo sellonlatbox,-128,-116,44,53 -monmax $directory $directoryNew # subsets and computes monthly max
                echo "cdo sellonlatbox,-128,-116,44,53 -monmax $directory $directoryNew # subsets and computes monthly max"
        done
done



echo "### Ending at: $(date) ###"
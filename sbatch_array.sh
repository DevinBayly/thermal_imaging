#!/bin/bash
#SBATCH --output=Sample_SLURM_Job-%a.out
#SBATCH --ntasks=16
#SBATCH --nodes=1             
#SBATCH --mem=5gb                    
#SBATCH --time=01:30:00   
#SBATCH --partition=standard
#SBATCH --account=visteam   
#SBATCH --array 0-5
 
# SLURM Inherits your environment. cd $SLURM_SUBMIT_DIR not needed
pwd; hostname; date
cd /groups/chrisreidy/baylyd/thermal_imaging/ 
singularity exec thermal_imaging.sif bash process_vids.sh $1 $2 ${SLURM_ARRAY_TASK_ID}

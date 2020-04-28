## this must be run from codes directory, --workdir is not used from here
#!/usr/bin/sh

#SBATCH --job-name=26_protein_project
#SBATCH --qos=csqos
##SBATCH --workdir=/scratch/akabir4/protein_project_1
#SBATCH --output=/scratch/akabir4/protein_project_1/outputs/log_26-%N-%j.output
#SBATCH --error=/scratch/akabir4/protein_project_1/outputs/log_26-%N-%j.error
#SBATCH --mail-user=<akabir4@gmu.edu>
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --gres=gpu:1
#SBATCH --partition=gpuq
#SBATCH --mem=64G


##this is the first testing file.  the path should be started from the "--workdir"
##--workdir is not used in this file. While running this job, mount into codes directory.
python full_run.py

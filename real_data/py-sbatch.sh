#!/bin/bash

###
# Parameters for sbatch
#
NUM_NODES=1
NUM_CORES=4
NUM_GPUS=0
JOB_NAME="python"

###
# Conda parameters
#
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=decongestion_env

sbatch \
	-N $NUM_NODES \
	-c $NUM_CORES \
	--gres=gpu:$NUM_GPUS \
	--job-name $JOB_NAME \
	-o 'slrum_runs/slurm-%N-%j.out' \
<<EOF
#!/bin/bash
echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"

# Setup the conda env
echo "*** Activating environment $CONDA_ENV ***"
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Run python with the args to the script
python3 $@

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF


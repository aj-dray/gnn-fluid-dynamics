#!/bin/bash


#! DIRECTIVES

#SBATCH -J train
#SBATCH -A T2-CS181-GPU  #NIKIFORAKIS-DRAY-SL2-GPU  #
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1      # should not exceed number of GPUS
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#! #SBATCH --no-requeue
#SBATCH --output=/home/ajd246/code/project/slurm_log/train/%j.txt
#! #SBATCH --dependency=afterok:9922345


#! VARIABLES

workdir="/home/ajd246/code/project-exec"
application="./scripts/train.sh"
options="$1"       # first commandline argument is name of config



#! ENVIRONEMNT

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment


#! PARALLEL SETUP

numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
export OMP_NUM_THREADS=1
np=$[${numnodes}*${mpi_tasks_per_node}]


#! EXEC

CMD="$application $options"
#CMD="mpirun -npernode $mpi_tasks_per_node -np $np $application $options"

cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

#!/bin/bash
#SBATCH -A MYACCOUNT
#SBATCH -p skylake
#SBATCH -N 2
#SBATCH -n 64
#SBATCH -t 02:00:00

module purge
module load rhel7/default-basic
# binutils and gcc modules only required if building a custom solver/application
# module load binutils-2.28-gcc-5.4.0-bqtph6o
# module load gcc-5.4.0-gcc-4.8.5-fis24gg

module load libpciaccess-0.13.5-gcc-5.4.0-avw7thq
module load xz-5.2.3-gcc-5.4.0-gavil4p
module load zlib-1.2.11-gcc-5.4.0-dmjwhms
module load libxml2-2.9.4-gcc-5.4.0-h6dkfhd
module load hwloc-1.11.7-gcc-5.4.0-ffzjos5
module load openmpi-2.1.1-gcc-5.4.0-pt4josx
module load scotch-6.0.4-gcc-5.4.0-dol5gkh
module load openfoam-org-5.0-gcc-5.4.0-xcufi7x
#    module load openfoam-org-2.4.0-gcc-5.4.0-nurvyom


# This is required to setup the proper variables and library paths
. $WM_PROJECT_DIR/etc/bashrc

# Decompose case to number of processors used, this is defined in system/decomposeParDict
decomposePar

mpirun -n $SLURM_NTASKS -N $SLURM_NTASKS_PER_NODE simpleFoam -parallel

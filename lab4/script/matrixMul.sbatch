#!/bin/bash
#SBATCH -J lab4         # Job name
#SBATCH -p compclass    # Queue name (or "compclass_unstable", or "gpuserv", or "a100serv")
#SBATCH -o lab4.%j.out  # Name of stdout output file (%j expands to %jobId)
#SBATCH -N 1            # Total number of nodes requested
#SBATCH -n 2            # Total number of mpi tasks requested
#SBATCH -t 00:10:00     # Run time (hh:mm:ss) - 1 minute

module load mpi/mpich-x86_64

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dimension X size> <dimension Y size>"
    exit 1
fi

dimX=$1
dimY=$2

SIZE=$((dimX * dimY))

mpicxx cart.cpp -o cart

echo -e "CART: $dimX x $dimY\n"

mpirun -np $SIZE ./cart $dimX $dimY

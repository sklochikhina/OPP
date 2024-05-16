#!/bin/bash
#SBATCH -J lab3         # Job name
#SBATCH -p compclass    # Queue name (or "compclass_unstable", or "gpuserv", or "a100serv")
#SBATCH -o lab3.%j.out  # Name of stdout output file (%j expands to %jobId)
#SBATCH -N 1            # Total number of nodes requested
#SBATCH -n 1            # Total number of mpi tasks requested
#SBATCH -t 00:10:00     # Run time (hh:mm:ss) - 1 minute

module load mpi/intelmpi

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <num_proc>"
    exit 1
fi

MPI_NUM_PROC=$1

mpicxx slauOMP.cpp -o slauOMP

echo -e "\n1st run:\n"
mpirun -np "$MPI_NUM_PROC" ./slauOMP

echo -e "\n2nd run:\n"
mpirun -np "$MPI_NUM_PROC" ./slauOMP

echo -e "\n3rd run:\n"
mpirun -np "$MPI_NUM_PROC" ./slauOMP
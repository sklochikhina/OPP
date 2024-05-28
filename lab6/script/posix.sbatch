#!/bin/bash
#SBATCH -J lab5 	# Job name
#SBATCH -p compclass  	# Queue name (or "compclass_unstable", or "gpuserv", or "a100serv")
#SBATCH -o lab6.%j.out 	# Name of stdout output file (%j expands to %jobId)
#SBATCH -e lab6.%j.err 	# Name of stderr output file (%j expands to %jobId)
#SBATCH -N 2 		# Total number of nodes requested
#SBATCH -n 2 		# Total number of mpi tasks requested
#SBATCH -t 00:10:00 	# Run time (hh:mm:ss)

module load mpi/mpich-x86_64

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <num_processes>"
    exit 1
fi

mpicxx -pthread posix.cpp -o posix

echo -e "SIZE = $1\n"
mpirun -np $1 ./posix

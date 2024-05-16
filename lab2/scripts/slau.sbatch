#!/bin/bash
#SBATCH -J lab2.1       # Job name
#SBATCH -p compclass    # Queue name (or "compclass_unstable", or "gpuserv", or "a100serv")
#SBATCH -o lab3.%j.out  # Name of stdout output file (%j expands to %jobId)
#SBATCH -N 1            # Total number of nodes requested
#SBATCH -n 1            # Total number of mpi tasks requested
#SBATCH -t 00:10:00     # Run time (hh:mm:ss) - 1 minute

g++ slauOMP.cpp -o slauOMP

echo -e "\n1st run:\n"
./slauOMP

echo -e "\n2nd run:\n"
./slauOMP

echo -e "\n3rd run:\n"
./slauOMP
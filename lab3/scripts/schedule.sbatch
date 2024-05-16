#!/bin/bash
#SBATCH -J lab3                 # Job name
#SBATCH -p compclass            # Queue name (or "compclass_unstable", or "gpuserv", or "a100serv")
#SBATCH -o lab3.schedule.%j.out # Name of stdout output file (%j expands to %jobId)
#SBATCH -e lab3.schedule.%j.err # Name of stderr output file (%j expands to %jobId)
#SBATCH -N 1                    # Total number of nodes requested
#SBATCH -n 1                    # Total number of mpi tasks requested
#SBATCH -t 00:30:00             # Run time (hh:mm:ss)

module load mpi/intelmpi

export OMP_NUM_THREADS=4

g++ -fopenmp schedule.cpp -o schedule

params=(1 2 3 4 5 6 7 8 9 10 20 50 80 100 200 500 1000 1500 2000)

for i in "${params[@]}"; do
    echo -e "\nRunning with OMP_SCHEDULE = static,$i"
    export OMP_SCHEDULE=static,$i
    ./schedule

    echo -e "\nRunning with OMP_SCHEDULE = dynamic,$i"
    export OMP_SCHEDULE=dynamic,$i
    ./schedule

    echo -e "\nRunning with OMP_SCHEDULE = guided,$i"
    export OMP_SCHEDULE=guided,$i
    ./schedule
done
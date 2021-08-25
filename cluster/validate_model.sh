#!/bin/bash
#PBS -l select=4:ncpus=15:mem=5GB
#PBS -l walltime=6:0:0
#PBS -q short_cpuQ
#PBS -M giovanni.detoni@unitn.it
#PBS -V
#PBS -m be

cd learning_programs_with_arguments

mkdir -p $save_result_dir

cd ./validation/

export PYTHONPATH=../

bash validate_quicksort_model.sh $filename $operations $save_result_dir

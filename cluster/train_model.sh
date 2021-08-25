#!/bin/bash
#PBS -l select=4:ncpus=15:mem=5GB
#PBS -l walltime=72:0:0
#PBS -q common_cpuQ
#PBS -M giovanni.detoni@unitn.it
#PBS -V
#PBS -m be

# Strict bash mode
# Disabled since it does not work well
# on the cluster environment
#set -euo pipefail
#IFS=$'\n\t'

CPUS=60

cd learning_programs_with_arguments

cd ./trainings/

export PYTHONPATH=../

if [ ${command_set} != "none" ]; then
  if [ ${use_complete_actions} != "none" ]; then
    if [ ${widen} != "none" ]; then
      if [ ${use_gpu} != "none" ]; then
        python3 train_quicksorting.py --seed=${seed} --tensorboard --verbose --save-model  \
          --num-cpus ${CPUS} --tb-base-dir ${output_dir_tb} --penalize-level-0 --keep-training --sample-error-prob ${train_errors} \
          --dir-noise ${dirichlet_noise} --dir-eps ${dirichlet_eps} --do-not-expose-pointers-values --normalize-policy \
          --model-dir ${output_model_dir} --check-autograd-from ${start_nancheck} --check-autograd-to ${end_nancheck} \
          --max-train-level ${max_train_level} --load-model ${load_model_path} --start-level ${start_level} --use-learned-hyper \
          --set-max-exploration-nodes ${max_exploration_nodes} --default-childs ${default_childs} --widening --gpu --save-counts \
          ${command_set} ${use_complete_actions}
      else
        python3 train_quicksorting.py --seed=${seed} --tensorboard --verbose --save-model  \
          --num-cpus ${CPUS} --tb-base-dir ${output_dir_tb} --penalize-level-0 --keep-training --sample-error-prob ${train_errors} \
          --dir-noise ${dirichlet_noise} --dir-eps ${dirichlet_eps} --do-not-expose-pointers-values --normalize-policy \
          --model-dir ${output_model_dir} --check-autograd-from ${start_nancheck} --check-autograd-to ${end_nancheck} \
          --max-train-level ${max_train_level} --load-model ${load_model_path} --start-level ${start_level} --use-learned-hyper \
          --set-max-exploration-nodes ${max_exploration_nodes} --default-childs ${default_childs} --widening --save-counts\
          ${command_set} ${use_complete_actions}
      fi
    else
      if [ ${use_gpu} != "none" ]; then
        python3 train_quicksorting.py --seed=${seed} --tensorboard --verbose --save-model  \
          --num-cpus ${CPUS} --tb-base-dir ${output_dir_tb} --penalize-level-0 --keep-training --sample-error-prob ${train_errors} \
          --dir-noise ${dirichlet_noise} --dir-eps ${dirichlet_eps} --do-not-expose-pointers-values --normalize-policy \
          --model-dir ${output_model_dir} --check-autograd-from ${start_nancheck} --check-autograd-to ${end_nancheck} \
          --max-train-level ${max_train_level} --load-model ${load_model_path} --start-level ${start_level} --use-learned-hyper \
          --set-max-exploration-nodes ${max_exploration_nodes} --default-childs ${default_childs} --gpu --save-counts \
          ${command_set} ${use_complete_actions}
      else
        python3 train_quicksorting.py --seed=${seed} --tensorboard --verbose --save-model  \
          --num-cpus ${CPUS} --tb-base-dir ${output_dir_tb} --penalize-level-0 --keep-training --sample-error-prob ${train_errors} \
          --dir-noise ${dirichlet_noise} --dir-eps ${dirichlet_eps} --do-not-expose-pointers-values --normalize-policy \
          --model-dir ${output_model_dir} --check-autograd-from ${start_nancheck} --check-autograd-to ${end_nancheck} \
          --max-train-level ${max_train_level} --load-model ${load_model_path} --start-level ${start_level} --use-learned-hyper --save-counts \
          --set-max-exploration-nodes ${max_exploration_nodes} --default-childs ${default_childs} \
          ${command_set} ${use_complete_actions}
      fi
    fi
  else
    if [ ${widen} != "none" ]; then
      if [ ${use_gpu} != "none" ]; then
      python3 train_quicksorting.py --seed=${seed} --tensorboard --verbose --save-model  \
        --num-cpus ${CPUS} --tb-base-dir ${output_dir_tb} --penalize-level-0 --keep-training --sample-error-prob ${train_errors} \
        --dir-noise ${dirichlet_noise} --dir-eps ${dirichlet_eps} --do-not-expose-pointers-values --normalize-policy \
        --model-dir ${output_model_dir} --check-autograd-from ${start_nancheck} --check-autograd-to ${end_nancheck} \
        --max-train-level ${max_train_level} --load-model ${load_model_path} --start-level ${start_level} --use-learned-hyper \
        --set-max-exploration-nodes ${max_exploration_nodes} --default-childs ${default_childs} --widening --gpu \
        ${command_set}
    else
      python3 train_quicksorting.py --seed=${seed} --tensorboard --verbose --save-model  \
        --num-cpus ${CPUS} --tb-base-dir ${output_dir_tb} --penalize-level-0 --keep-training --sample-error-prob ${train_errors} \
        --dir-noise ${dirichlet_noise} --dir-eps ${dirichlet_eps} --do-not-expose-pointers-values --normalize-policy \
        --model-dir ${output_model_dir} --check-autograd-from ${start_nancheck} --check-autograd-to ${end_nancheck} \
        --max-train-level ${max_train_level} --load-model ${load_model_path} --start-level ${start_level} --use-learned-hyper \
        --set-max-exploration-nodes ${max_exploration_nodes} --default-childs ${default_childs} --widening --save-counts\
        ${command_set}
    fi
    else
      if [ ${use_gpu} != "none" ]; then
        python3 train_quicksorting.py --seed=${seed} --tensorboard --verbose --save-model  \
          --num-cpus ${CPUS} --tb-base-dir ${output_dir_tb} --penalize-level-0 --keep-training --sample-error-prob ${train_errors} \
          --dir-noise ${dirichlet_noise} --dir-eps ${dirichlet_eps} --do-not-expose-pointers-values --normalize-policy \
          --model-dir ${output_model_dir} --check-autograd-from ${start_nancheck} --check-autograd-to ${end_nancheck} \
          --max-train-level ${max_train_level} --load-model ${load_model_path} --start-level ${start_level} --use-learned-hyper \
          --set-max-exploration-nodes ${max_exploration_nodes} --default-childs ${default_childs} --gpu --save-counts \
          ${command_set}
      else
        python3 train_quicksorting.py --seed=${seed} --tensorboard --verbose --save-model  \
          --num-cpus ${CPUS} --tb-base-dir ${output_dir_tb} --penalize-level-0 --keep-training --sample-error-prob ${train_errors} \
          --dir-noise ${dirichlet_noise} --dir-eps ${dirichlet_eps} --do-not-expose-pointers-values --normalize-policy \
          --model-dir ${output_model_dir} --check-autograd-from ${start_nancheck} --check-autograd-to ${end_nancheck} \
          --max-train-level ${max_train_level} --load-model ${load_model_path} --start-level ${start_level} --use-learned-hyper \
          --set-max-exploration-nodes ${max_exploration_nodes} --default-childs ${default_childs} --save-counts \
          ${command_set}
      fi
    fi
  fi
else
  if [ ${use_complete_actions} != "none" ]; then
    if [ ${use_gpu} != "none" ]; then
      python3 train_quicksorting.py --seed=${seed} --tensorboard --verbose --save-model  \
      --num-cpus ${CPUS} --tb-base-dir ${output_dir_tb} --penalize-level-0 --keep-training --sample-error-prob ${train_errors} \
      --dir-noise ${dirichlet_noise} --dir-eps ${dirichlet_eps} --do-not-expose-pointers-values --normalize-policy \
      --model-dir ${output_model_dir} --check-autograd-from ${start_nancheck} --check-autograd-to ${end_nancheck} \
      --max-train-level ${max_train_level} --load-model ${load_model_path} --start-level ${start_level} --use-learned-hyper \
      --set-max-exploration-nodes ${max_exploration_nodes} --default-childs ${default_childs} --widening --gpu --save-counts \
      ${use_complete_actions}
    else
      python3 train_quicksorting.py --seed=${seed} --tensorboard --verbose --save-model  \
      --num-cpus ${CPUS} --tb-base-dir ${output_dir_tb} --penalize-level-0 --keep-training --sample-error-prob ${train_errors} \
      --dir-noise ${dirichlet_noise} --dir-eps ${dirichlet_eps} --do-not-expose-pointers-values --normalize-policy \
      --model-dir ${output_model_dir} --check-autograd-from ${start_nancheck} --check-autograd-to ${end_nancheck} \
      --max-train-level ${max_train_level} --load-model ${load_model_path} --start-level ${start_level} --use-learned-hyper \
      --set-max-exploration-nodes ${max_exploration_nodes} --default-childs ${default_childs} --widening --save-counts\
      ${use_complete_actions}
    fi
  else
    if [ ${use_gpu} != "none" ]; then
      python3 train_quicksorting.py --seed=${seed} --tensorboard --verbose --save-model \
      --num-cpus ${CPUS} --tb-base-dir ${output_dir_tb} --penalize-level-0 --keep-training --sample-error-prob ${train_errors} \
      --dir-noise ${dirichlet_noise} --dir-eps ${dirichlet_eps} --do-not-expose-pointers-values --normalize-policy \
      --model-dir ${output_model_dir} --check-autograd-from ${start_nancheck} --check-autograd-to ${end_nancheck} \
      --max-train-level ${max_train_level} --load-model ${load_model_path} --start-level ${start_level} --use-learned-hyper \
      --set-max-exploration-nodes ${max_exploration_nodes} --default-childs ${default_childs} --widening --gpu --save-counts
    else
      python3 train_quicksorting.py --seed=${seed} --tensorboard --verbose --save-model \
      --num-cpus ${CPUS} --tb-base-dir ${output_dir_tb} --penalize-level-0 --keep-training --sample-error-prob ${train_errors} \
      --dir-noise ${dirichlet_noise} --dir-eps ${dirichlet_eps} --do-not-expose-pointers-values --normalize-policy \
      --model-dir ${output_model_dir} --check-autograd-from ${start_nancheck} --check-autograd-to ${end_nancheck} \
      --max-train-level ${max_train_level} --load-model ${load_model_path} --start-level ${start_level} --use-learned-hyper \
      --set-max-exploration-nodes ${max_exploration_nodes} --default-childs ${default_childs} --widening --save-counts
    fi
  fi
fi

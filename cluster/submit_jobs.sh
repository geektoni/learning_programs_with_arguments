#!/bin/bash

export command_set="none"
if [ ${1} == "complete" ]; then
  export command_set="none"
elif [ ${1} == "without-partition-update" ]; then
  export command_set="--without-partition-update"
elif [ ${1} == "without-save-load-partition" ]; then
  export command_set="--without-save-load-partition"
elif [ ${1} == "recursive" ]; then
  export command_set="--recursive-quicksort"
else
  export command_set="--reduced-operation-set"
fi

export train_errors=${2}

export output_dir_tb=${3}
export seed=${4}

export dirichlet_noise=${5}

export use_complete_actions=""
if [ ${6} == "True" ]; then
  export use_complete_actions="--use-complete-actions"
else
  export use_complete_actions="none"
fi

export dirichlet_eps=${7}

export output_model_dir=${8}

export start_nancheck=${9}

export end_nancheck=${10}

export max_train_level=${11}

export result_name=${1}-${2}-${4}-${5}-${6}-${7}

export load_model_path=${12}

export start_level=${13}

export max_exploration_nodes=${14}

export default_childs=${15}

export use_gpu=""
if [ ${16} == "True" ]; then
  export use_gpu="--gpu"
else
  export use_gpu="none"
fi

export widen=""
if [ ${17} == "True" ]; then
  export widen="--widening"
else
  export widen="none"
fi


qsub -V -N "$result_name" train_model.sh
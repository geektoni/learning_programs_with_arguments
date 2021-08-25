from environments.quicksort_list_env import QuickSortListEnv, ListEnvEncoder
from core.policy import Policy
import core.config as conf
import torch
from core.mcts import MCTS
from visualization.visualise_mcts import MCTSvisualiser
from visualization.visualise_automa import VisualizeAutoma

import os

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

import numpy as np
import pandas as pd

import argparse

np.random.seed(42)
torch.manual_seed(42)

if __name__ == "__main__":

    # Path to load policy
    #load_path = '../models/list_npi_2019_5_16-10_19_59-1.pth'
    default_load_path = '../models/list_npi_2019_5_13-9_26_38-1.pth'

    # Get command line params
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help='random seed', default=np.random.randint(0, 100000), type=int)
    parser.add_argument("--load-path", help='path to model to validate', default=default_load_path)
    parser.add_argument('--verbose', help='print training monitoring in console', action='store_true')
    parser.add_argument('--save-results', help='save training progress in .txt file', action='store_true')
    parser.add_argument('--num-cpus', help='number of cpus to use', default=8, type=int)
    parser.add_argument('--min-length', help='Minimum size of the list we want to order', default=2, type=int)
    parser.add_argument('--max-length', help='Max size of the list we want to order', default=7, type=int)
    parser.add_argument('--validation-length', help='Size of the validation lists we want to order', default=7,
                        type=int)
    parser.add_argument('--program', help='Size of the validation lists we want to order', default='QUICKSORT',
                        type=str)
    parser.add_argument('--iter', help="Number of iterations", default=1000, type=int)
    parser.add_argument('--skip-errors', help='save training progress in .txt file', action='store_false', default=True)

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Obtain the various configuration options from the name.
    load_path = args.load_path
    filename = os.path.split(load_path)[1]
    values = filename.split("-")

    date = values[0]
    time_ = values[1]
    seed = values[2]
    str_c = values[3].lower() == "true"
    pen_level_0 = values[4].lower() == "true"
    leve_0_pen = float(values[5])
    expose_stack = values[6].lower() == "true"
    samp_err_poss = float(values[7])
    without_p_upd = values[8].lower() == "true"
    reduced_op_set = values[9].lower() == "true"
    keep_training = values[10].lower() == "true"
    recursive_quicksort = values[11].lower() == "true"
    do_not_expose_pointer_values = values[12].lower() == "true"
    complete_actions = values[13].lower() == "true"
    dir_noise = float(values[14])
    dir_eps = float(values[15])
    normalize_policies = values[16].split(".")[0].lower() == "true"

    # Load environment constants
    env_tmp = QuickSortListEnv(length=5, encoding_dim=conf.encoding_dim, expose_stack=expose_stack,
                               without_partition_update=without_p_upd, sample_from_errors_prob=samp_err_poss,
                               reduced_set=reduced_op_set, recursive_version=recursive_quicksort,
                               expose_pointers_value=do_not_expose_pointer_values,
                               complete_programs=complete_actions)
    num_programs = env_tmp.get_num_programs()
    num_non_primary_programs = env_tmp.get_num_non_primary_programs()
    observation_dim = env_tmp.get_observation_dim()
    programs_library = env_tmp.programs_library

    # Load Alpha-NPI policy
    encoder = ListEnvEncoder(env_tmp.get_observation_dim(), conf.encoding_dim)
    indices_non_primary_programs = [p['index'] for _, p in programs_library.items() if p['level'] > 0]
    policy = Policy(encoder, conf.hidden_size, num_programs, num_non_primary_programs, conf.program_embedding_dim,
                    conf.encoding_dim, indices_non_primary_programs, conf.learning_rate)

    policy.load_state_dict(torch.load(args.load_path))

    # Prepare mcts params
    length = 7

    if without_p_upd:
        max_depth_dict = {1: 3 * (length - 1) + 2, 2: 4, 3: 4, 4: length + 2}
    elif reduced_op_set:
        max_depth_dict = {1: 3 * (length - 1) + 2, 2: 6, 3: length + 2}
    elif recursive_quicksort:
        max_depth_dict = {1: 3 * (length - 1) + 2, 2: 7, 3: 3}
    else:
        max_depth_dict = {1: 3, 2: 2 * (length - 1) + 2, 3: 4, 4: 4, 5: length + 2}

    mcts_test_params = {'number_of_simulations': conf.number_of_simulations_for_validation,
                        'max_depth_dict': max_depth_dict, 'temperature': conf.temperature,
                        'c_puct': conf.c_puct, 'exploit': True, 'level_closeness_coeff': conf.level_closeness_coeff,
                        'gamma': conf.gamma, "penalize_level_0": pen_level_0, 'use_structural_constraint': False,
                        'max_recursion_depth': length*2+1,'use_arguments': (not complete_actions),
                        'normalize_policy': normalize_policies}

    # Start debugging ...
    env = QuickSortListEnv(length=length, encoding_dim=conf.encoding_dim, expose_stack=expose_stack,
                               without_partition_update=without_p_upd, sample_from_errors_prob=samp_err_poss,
                               reduced_set=reduced_op_set, recursive_version=recursive_quicksort,
                                expose_pointers_value=do_not_expose_pointer_values,
                           complete_programs=complete_actions)
    program_index = env.programs_library[args.program]['index']

    total_reward = []
    total_failed_programs = [0 for a in range(0, len(env.programs_library))]
    total_failed_state_index = [[] for a in range(0, len(env.programs_library))]
    total_failures = 0

    automata = VisualizeAutoma(env)

    for i in tqdm(range(args.iter)):
        env = QuickSortListEnv(length=length, encoding_dim=conf.encoding_dim, expose_stack=expose_stack,
                               without_partition_update=without_p_upd, sample_from_errors_prob=samp_err_poss,
                               reduced_set=reduced_op_set, recursive_version=recursive_quicksort,
                               expose_pointers_value=do_not_expose_pointer_values,
                               complete_programs=complete_actions)
        mcts = MCTS(policy, env, program_index, **mcts_test_params)
        res = mcts.sample_execution_trace()
        root_node, r, failed_state_index = res[6], res[7], res[12]

        # Store the points
        if r != -1:
            automata.add(policy.encoder, root_node)

        for j in range(0, len(failed_state_index)):
            total_failed_state_index[j] += failed_state_index[j]

        if len(mcts.programs_failed_indices) != 0:
            total_failed_programs[mcts.programs_failed_indices[len(mcts.programs_failed_indices)-1]] += 1
        else:
            if r == -1:
                total_failures += 1

        total_reward.append(1 if r > -1 else 0)

    # Reduce dimensionality of points
    automata.compute()
    automata.plot()



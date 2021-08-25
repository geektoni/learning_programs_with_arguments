from environments.quicksort_list_env import ListEnvEncoder, QuickSortListEnv
from core.curriculum import CurriculumScheduler
from core.policy import Policy
import core.config as conf
from core.trainer import Trainer
from core.prioritized_replay_buffer import PrioritizedReplayBuffer
import argparse
import numpy as np
import torch
from tensorboardX import SummaryWriter
import time

if __name__ == "__main__":

    # Get command line params
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help='random seed', default=1, type=int)
    parser.add_argument('--tensorboard', help='display on tensorboard', action='store_true')
    parser.add_argument('--verbose', help='print training monitoring in console', action='store_true')
    parser.add_argument('--save-model', help='save neural network model', action='store_true')
    parser.add_argument('--save-results', help='save training progress in .txt file', action='store_true')
    parser.add_argument('--num-cpus', help='number of cpus to use', default=8, type=int)
    parser.add_argument('--load-model', help='Load a pretrained model and train from there', default="", type=str)
    parser.add_argument('--min-length', help='Minimum size of the list we want to order', default=2, type=int)
    parser.add_argument('--max-length', help='Max size of the list we want to order', default=7, type=int)
    parser.add_argument('--validation-length', help='Size of the validation lists we want to order', default=7, type=int)
    parser.add_argument('--start-level', help='Specify up to which level we are trying to learn', default=1, type=int)
    parser.add_argument('--tb-base-dir', help='Specify base tensorboard dir', default="runs", type=str)
    parser.add_argument('--structural-constraint', help="Use the structural constraint to train", action='store_true')
    parser.add_argument('--gamma', help="Specify gamma discount factor", default=0.97, type=float)
    parser.add_argument('--penalize-level-0', help="Penalize level 0 operations when computing the Q-value", default=True, action='store_false')
    parser.add_argument('--level-0-penalty', help="Custom penalty value for the level 0 actions", default=1.0, type=float)
    parser.add_argument('--expose-stack', help="When observing the environment, simply expose the firs two element of the stack", default=False, action='store_true')
    parser.add_argument('--sample-error-prob', help="Probability of sampling error envs when doing training", default=0.3, type=float)
    parser.add_argument('--without-partition-update', help="Train everything without the partition update program", default=False, action="store_true")
    parser.add_argument('--without-save-load-partition', help="Train everything without the save load partition program", default=False, action="store_true")
    parser.add_argument('--reduced-operation-set', help="Train everything with a reduced set of operations", default=False, action="store_true")
    parser.add_argument('--keep-training', help="Keep training even if we reach 'perfection' on all the task", default=False, action="store_true")
    parser.add_argument('--recursive-quicksort', help="The QUICKSORT function is made recursive.", default=False, action="store_true")
    parser.add_argument('--max-rec-depth', help="Maximum recursion depth", default=100, type=int)
    parser.add_argument('--max-quicksort-depth', help="Maximum recursion depth", default=15, type=int)
    parser.add_argument('--do-not-expose-pointers-values', help="Do not expose pointers values in the observations", action="store_false", default=True)
    parser.add_argument('--use-complete-actions', help="Train over the complete set of actions", action="store_true", default=False)
    parser.add_argument('--dir-noise', help="Specify amount of dirichlet noise", default=0.03, type=float)
    parser.add_argument('--dir-eps', help="Specify the importance of exploration", default=0.35, type=float)
    parser.add_argument('--normalize-policy', help="Normalize police once we do the tree expansion", default=False, action="store_true")
    parser.add_argument('--check-autograd-from', help="Start checking for nan when doing autograd", default=-1, type=int)
    parser.add_argument('--check-autograd-to', help="End checking for nan when doing autograd", default=100,
                        type=int)
    parser.add_argument('--model-dir', help="Location were to save the trained models", default="../models", type=str)
    parser.add_argument('--max-train-level', help="Train the model up to a given level", default=-1, type=int)
    parser.add_argument('--use-learned-hyper', help="Use specific hyperparameter for each action", default=False, action="store_true")
    parser.add_argument('--set-max-exploration-nodes', help="Specify the maximum amount of nodes we can expand", default=1.0, type=float)
    parser.add_argument('--widening', help="Use the sampling version of the MCTS", default=False, action="store_true")
    parser.add_argument('--default-childs', help="How many nodes to sample when using --widening", default=10, type=int)
    parser.add_argument('--gpu', help="Use GPU if available", default=False, action="store_true")
    parser.add_argument('--save-counts', help="Save expanded nodes counts", default=False, action="store_true")


    args = parser.parse_args()

    # Get arguments
    seed = args.seed
    tensorboard = args.tensorboard
    base_tb_dir = args.tb_base_dir
    verbose = args.verbose
    save_model = args.save_model
    save_results = args.save_results
    num_cpus = args.num_cpus
    sample_error_prob = args.sample_error_prob
    conf.gamma = args.gamma
    conf.penalize_level_0 = args.penalize_level_0
    conf.level_0_penalty = args.level_0_penalty
    expose_pointers_values = args.do_not_expose_pointers_values
    complete_actions = args.use_complete_actions
    default_childs = args.default_childs

    use_gpu = args.gpu

    autograd_from = args.check_autograd_from
    autograd_to = args.check_autograd_to

    max_train_level = args.max_train_level

    used_learned_hyper = args.use_learned_hyper

    max_exploration_nodes = args.set_max_exploration_nodes

    save_node_counts = args.save_counts

    # Verbose output
    if verbose:
        print(args)

    load_model = False
    if args.load_model != "" and args.load_model != "none":
        load_model = True

    custom_start_level = False
    if args.start_level != 0:
        custom_start_level = True

    # Set if we are using the structural constraint
    if args.structural_constraint:
        conf.structural_constraint = True

    # Set number of cpus used
    torch.set_num_threads(num_cpus)

    # get date and time
    ts = time.localtime(time.time())
    date_time = '{}_{}_{}-{}_{}_{}'.format(ts[0], ts[1], ts[2], ts[3], ts[4], ts[5])
    # Path to save policy
    model_save_path = '{}/list_npi_{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}.pth'.format(
        args.model_dir,
        date_time, seed,
        args.structural_constraint,
        args.penalize_level_0,
        args.level_0_penalty,
        args.expose_stack,
        sample_error_prob,
        args.without_partition_update,
        args.reduced_operation_set,
        args.keep_training,
        args.recursive_quicksort,
        expose_pointers_values,
        complete_actions,
        args.dir_noise,
        args.dir_eps,
        args.normalize_policy,
        args.without_save_load_partition,
        args.default_childs,
        args.widening,
        use_gpu)
    # Path to save results
    results_save_path = '../results/list_npi_{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}.txt'.format(
        date_time, seed, args.structural_constraint,
        args.penalize_level_0, args.level_0_penalty, args.expose_stack,
        sample_error_prob, args.without_partition_update,
        args.reduced_operation_set,
        args.keep_training,
        args.recursive_quicksort,
        expose_pointers_values,
        complete_actions,
        args.dir_noise,
        args.dir_eps,
        args.normalize_policy,
        args.without_save_load_partition,
        args.default_childs,
        args.widening,
        use_gpu)
    # Path to tensorboard
    tensorboard_path = '{}/list_npi_{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
        base_tb_dir,
        date_time, seed,
        args.structural_constraint,
        args.penalize_level_0,
        args.level_0_penalty,
        args.expose_stack,
        sample_error_prob,
        args.without_partition_update,
        args.reduced_operation_set,
        args.keep_training,
        args.recursive_quicksort,
        expose_pointers_values,
        complete_actions,
        args.dir_noise,
        args.dir_eps,
        args.normalize_policy,
        args.without_save_load_partition,
        args.default_childs,
        args.widening,
        use_gpu)

    total_nodes_save_path = '../results/list_npi_{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-nodes.txt'.format(
        date_time, seed, args.structural_constraint,
        args.penalize_level_0, args.level_0_penalty, args.expose_stack,
        sample_error_prob, args.without_partition_update,
        args.reduced_operation_set,
        args.keep_training,
        args.recursive_quicksort,
        expose_pointers_values,
        complete_actions,
        args.dir_noise,
        args.dir_eps,
        args.normalize_policy,
        args.without_save_load_partition,
        args.default_childs,
        args.widening,
        use_gpu)

    # Instantiate tensorboard writer
    if tensorboard:
        writer = SummaryWriter(tensorboard_path)

    # Instantiate file writer
    if save_results:
        results_file = open(results_save_path, 'w')

    # Instantiate the file used to keep all the nodes
    nodes_file = open(total_nodes_save_path, 'w')

    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load environment constants
    env_tmp = QuickSortListEnv(length=5, encoding_dim=conf.encoding_dim, expose_stack=args.expose_stack,
                               sample_from_errors_prob=sample_error_prob,
                               without_partition_update=args.without_partition_update,
                               without_save_load_partition=args.without_save_load_partition,
                               reduced_set=args.reduced_operation_set,
                               recursive_version=args.recursive_quicksort,
                               expose_pointers_value=expose_pointers_values,
                               complete_programs=complete_actions)
    num_programs = env_tmp.get_num_programs()
    num_non_primary_programs = env_tmp.get_num_non_primary_programs()
    observation_dim = env_tmp.get_observation_dim()
    programs_library = env_tmp.programs_library

    # Load alphanpi policy
    encoder = ListEnvEncoder(env_tmp.get_observation_dim(), conf.encoding_dim)
    indices_non_primary_programs = [p['index'] for _, p in programs_library.items() if p['level'] > 0]
    policy = Policy(encoder, conf.hidden_size, num_programs, num_non_primary_programs, conf.program_embedding_dim,
                    conf.encoding_dim, indices_non_primary_programs, conf.learning_rate, use_args=(not complete_actions),
                    use_gpu=use_gpu)

    # Load a pre-trained policy (to speed up testing)
    if load_model:
        policy.load_state_dict(torch.load(args.load_model))

    # Load replay buffer
    idx_tasks = [prog['index'] for key, prog in env_tmp.programs_library.items() if prog['level'] > 0]
    buffer = PrioritizedReplayBuffer(conf.buffer_max_length, idx_tasks, p1=conf.proba_replay_buffer)

    # Load curriculum sequencer
    curriculum_scheduler = CurriculumScheduler(conf.reward_threshold, num_non_primary_programs, programs_library,
                                               moving_average=0.99)
    curriculum_scheduler.maximum_level = args.start_level

    # Prepare mcts params
    length = 5

    if args.without_partition_update:
        max_depth_dict = {1: 3 * (length - 1) + 2, 2: 4, 3: 4, 4: length + 2}
    elif args.without_save_load_partition:
        max_depth_dict = {1: 3, 2: 2 * (length - 1) + 2, 3: 6, 4: length + 2}
    elif args.reduced_operation_set:
        max_depth_dict = {1: 3 * (length - 1) + 2, 2: 6, 3: length + 2}
    elif args.recursive_quicksort:
        max_depth_dict = {1: 3 * (length - 1) + 2, 2: 7, 3: 3}
    else:
        max_depth_dict = {1: 3, 2: 2 * (length - 1) + 2, 3: 4, 4: 4, 5: length + 2}

    mcts_train_params = {'number_of_simulations': conf.number_of_simulations, 'max_depth_dict': max_depth_dict,
                         'temperature': conf.temperature, 'c_puct': conf.c_puct, 'exploit': False,
                         'level_closeness_coeff': conf.level_closeness_coeff, 'gamma': conf.gamma,
                         'use_dirichlet_noise': True, 'use_structural_constraint': conf.structural_constraint,
                         'penalize_level_0': conf.penalize_level_0, 'level_0_penalty': conf.level_0_custom_penalty,
                         'max_recursion_depth': args.max_rec_depth, 'max_recursion_program_call': args.max_quicksort_depth,
                         'use_arguments': (not complete_actions), 'dir_epsilon': args.dir_eps, 'dir_noise': args.dir_noise,
                         'normalize_policy': args.normalize_policy, 'max_exploration_nodes': max_exploration_nodes,
                         'use_gpu': use_gpu}

    mcts_test_params = {'number_of_simulations': conf.number_of_simulations_for_validation,
                        'max_depth_dict': max_depth_dict, 'temperature': conf.temperature,
                        'c_puct': conf.c_puct, 'exploit': True, 'level_closeness_coeff': conf.level_closeness_coeff,
                        'gamma': conf.gamma, 'use_structural_constraint': conf.structural_constraint,
                        'penalize_level_0': conf.penalize_level_0, 'level_0_penalty': conf.level_0_custom_penalty,
                        'max_recursion_depth': args.max_rec_depth, 'max_recursion_program_call': args.max_quicksort_depth,
                        'use_arguments': (not complete_actions), 'dir_epsilon': args.dir_eps, 'dir_noise': args.dir_noise,
                        'normalize_policy': args.normalize_policy, 'max_exploration_nodes': max_exploration_nodes,
                        'use_gpu': use_gpu}

    if args.widening:
        mcts_train_params['default_childs'] = args.default_childs

    # Specify a custom start level
    if custom_start_level:
        curriculum_scheduler.maximum_level = args.start_level

    # Instanciate trainer
    trainer = Trainer(env_tmp, policy, buffer, curriculum_scheduler, mcts_train_params,
                      mcts_test_params, conf.num_validation_episodes, conf.num_episodes_per_task, conf.batch_size,
                      conf.num_updates_per_episode, verbose, autograd_from=autograd_from, autograd_to=autograd_to,
                      widening=args.widening)

    min_length = args.min_length
    max_length = args.max_length
    validation_length = args.validation_length
    failed_executions_envs = None

    # Start training
    for iteration in range(conf.num_iterations):
        # play one iteration
        task_index = curriculum_scheduler.get_next_task_index()
        task_level = env_tmp.get_program_level_from_index(task_index)
        length = np.random.randint(min_length, max_length+1)
        env = QuickSortListEnv(length=length, encoding_dim=conf.encoding_dim, expose_stack=args.expose_stack,
                               sample_from_errors_prob=sample_error_prob,
                               without_partition_update=args.without_partition_update,
                               without_save_load_partition=args.without_save_load_partition,
                               reduced_set=args.reduced_operation_set,
                               recursive_version=args.recursive_quicksort,
                               expose_pointers_value=expose_pointers_values,
                               complete_programs=complete_actions)

        if args.without_partition_update:
            max_depth_dict = {1: 3 * (length - 1) + 2, 2: 4, 3: 4, 4: length + 2}
        elif args.without_save_load_partition:
            max_depth_dict = {1: 3, 2: 2 * (length - 1) + 2, 3: 6, 4: length + 2}
        elif args.reduced_operation_set:
            max_depth_dict = {1: 3 * (length - 1) + 2, 2: 6, 3: length + 2}
        elif args.recursive_quicksort:
            max_depth_dict = {1: 3 * (length - 1) + 2, 2: 7, 3: 3}
        else:
            max_depth_dict = {1: 3, 2: 2 * (length - 1) + 2, 3: 4, 4: 4, 5: length + 2}

        # Restore the previous failed executions
        if failed_executions_envs != None:
            env.failed_executions_env = failed_executions_envs

        trainer.env = env
        trainer.mcts_train_params['max_depth_dict'] = max_depth_dict
        trainer.mcts_test_params['max_depth_dict'] = max_depth_dict

        # If we used learned hyperparameters, then we fetch them from the configuration file. This happens
        # only for programs which have a different hyper parameter specified.
        if used_learned_hyper:
            task_name = env.get_program_from_index(task_index)
            if conf.predefined_hyperparameters[task_name][0] != -1 \
                and conf.predefined_hyperparameters[task_name][1] != -1:
                trainer.mcts_train_params['dir_epsilon'] = conf.predefined_hyperparameters[task_name][0]
                trainer.mcts_train_params['dir_noise'] = conf.predefined_hyperparameters[task_name][1]
                trainer.mcts_test_params['dir_epsilon'] = conf.predefined_hyperparameters[task_name][0]
                trainer.mcts_test_params['dir_noise'] = conf.predefined_hyperparameters[task_name][1]
            else:
                trainer.mcts_train_params['dir_epsilon'] = args.dir_eps
                trainer.mcts_train_params['dir_noise'] = args.dir_noise
                trainer.mcts_test_params['dir_epsilon'] = args.dir_eps
                trainer.mcts_test_params['dir_noise'] = args.dir_noise


        # Play one iteration
        actor_loss, critic_loss, arguments_loss, total_nodes_training, total_selected_nodes_trainings = trainer.play_iteration(task_index)

        if tensorboard and not np.isnan(actor_loss):
            v_task_name = env.get_program_from_index(task_index)
            writer.add_scalar("loss/"+ v_task_name+"/actor", actor_loss, iteration)
            writer.add_scalar("loss/"+ v_task_name+"/value", critic_loss, iteration)
            writer.add_scalar("loss/"+ v_task_name+"/arguments", arguments_loss, iteration)

        # Save the failed execution env
        failed_executions_envs = env.failed_executions_env

        # perform validation
        complete_rewards = []
        if verbose:
            print("Start validation .....")
        for idx in curriculum_scheduler.get_tasks_of_maximum_level():
            task_level = env_tmp.get_program_level_from_index(idx)
            length = validation_length
            env = QuickSortListEnv(length=length, encoding_dim=conf.encoding_dim, expose_stack=args.expose_stack,
                                   validation_mode=True,
                                   without_partition_update=args.without_partition_update,
                                   without_save_load_partition=args.without_save_load_partition,
                                   reduced_set=args.reduced_operation_set,
                                   recursive_version=args.recursive_quicksort,
                                   expose_pointers_value=expose_pointers_values,
                                   complete_programs=complete_actions)

            if args.without_partition_update:
                max_depth_dict = {1: 3 * (length - 1) + 2, 2: 4, 3: 4, 4: length + 2}
            elif args.without_save_load_partition:
                max_depth_dict = {1: 3, 2: 2 * (length - 1) + 2, 3: 6, 4: length + 2}
            elif args.reduced_operation_set:
                max_depth_dict = {1: 3 * (length - 1) + 2, 2: 6, 3: length + 2}
            elif args.recursive_quicksort:
                max_depth_dict = {1: 3 * (length - 1) + 2, 2: 7, 3: 3}
            else:
                max_depth_dict = {1: 3, 2: 2 * (length - 1) + 2, 3: 4, 4: 4, 5: length + 2}

            trainer.env = env
            trainer.mcts_train_params['max_depth_dict'] = max_depth_dict
            trainer.mcts_test_params['max_depth_dict'] = max_depth_dict

            # If we used learned hyperparameters, then we fetch them from the configuration file. This happens
            # only for programs which have a different hyper parameter specified.
            if used_learned_hyper:
                task_name = env.get_program_from_index(task_index)
                if conf.predefined_hyperparameters[task_name][0] != -1 \
                        and conf.predefined_hyperparameters[task_name][1] != -1:
                    trainer.mcts_train_params['dir_epsilon'] = conf.predefined_hyperparameters[task_name][0]
                    trainer.mcts_train_params['dir_noise'] = conf.predefined_hyperparameters[task_name][1]
                    trainer.mcts_test_params['dir_epsilon'] = conf.predefined_hyperparameters[task_name][0]
                    trainer.mcts_test_params['dir_noise'] = conf.predefined_hyperparameters[task_name][1]
                else:
                    trainer.mcts_train_params['dir_epsilon'] = args.dir_eps
                    trainer.mcts_train_params['dir_noise'] = args.dir_noise
                    trainer.mcts_test_params['dir_epsilon'] = args.dir_eps
                    trainer.mcts_test_params['dir_noise'] = args.dir_noise

            # Evaluate performance on task idx
            v_rewards, v_lengths, programs_failed_indices = trainer.perform_validation_step(idx)
            # Update curriculum statistics
            curriculum_scheduler.update_statistics(idx, v_rewards)

            # If we specified a max training level, then we stick to it
            # and we do not advance
            if max_train_level != -1:
                if curriculum_scheduler.maximum_level > max_train_level:
                    curriculum_scheduler.maximum_level = max_train_level

            # Append the iteration, program index and reward
            for r in v_rewards:
                real_reward = 0 if r < 0 else 1
                complete_rewards.append((iteration,idx,real_reward))

        if save_node_counts:
            print(total_nodes_training, sum(total_selected_nodes_trainings))
            nodes_data = "{},".format(iteration)
            tmp = "{0:.4f},".format(sum(total_selected_nodes_trainings) / conf.num_episodes_per_task)
            for k in sorted(env.programs_library):
                index = int(env.programs_library[k]['index'])
                if index in total_nodes_training:
                    validation_data_idx = curriculum_scheduler.get_statistic(index)
                    nodes_data += "{}:{}-{}-".format(k, total_nodes_training[index], validation_data_idx)
            nodes_data += tmp
            nodes_data = nodes_data[0:-1]
            nodes_data += "\n"
            nodes_file.write(nodes_data)
            nodes_file.flush()

        # display training progress in tensorboard
        if tensorboard:
            for idx in curriculum_scheduler.get_tasks_of_maximum_level():
                v_task_name = env.get_program_from_index(idx)
                # record on tensorboard
                writer.add_scalar('validation/' + v_task_name, curriculum_scheduler.get_statistic(idx), iteration)

        # write training progress in txt file
        if save_results:
            for e in complete_rewards:
                reward_indication = "{},{},{}\n".format(e[0], e[1], e[2])
                results_file.write(reward_indication)
            #str = 'Iteration: {}'.format(iteration)
            #for idx in curriculum_scheduler.indices_non_primary_programs:
            #    task_name = env.get_program_from_index(idx)
            #    str += ', %s:%.3f' % (task_name, curriculum_scheduler.get_statistic(idx))
            #str += '\n'
            #results_file.write(str)

        # print new training statistics
        if verbose:
            curriculum_scheduler.print_statistics()
            print('')
            print('')

        # If succeed on al tasks, go directly to next list length
        if curriculum_scheduler.maximum_level > env.get_maximum_level():
            if not args.keep_training:
                break
            else:
                # keep on training
                curriculum_scheduler.maximum_level = env.get_maximum_level()

        # If we specified a max training level, then we stick to it
        # and we do not advance
        if max_train_level != -1:
            if curriculum_scheduler.maximum_level > max_train_level:
                curriculum_scheduler.maximum_level = max_train_level

        # Save policy
        if save_model:
            torch.save(policy.state_dict(), model_save_path)

        # Check if we reached the maximum amount of iterations. If this happens,
        # then exit
        current_program_name = env.get_program_from_index(task_index)
        if conf.max_training_iterations[current_program_name] != -1:
            if trainer.iterations_for_each_program[current_program_name] >= conf.max_training_iterations[current_program_name]:
                print("Reached the maximum amount of training iterations of {} for operation {}".format(
                    conf.max_training_iterations[current_program_name], current_program_name
                ))
                break


    # Close tensorboard writer
    if verbose:
        print('End of training !')
    if tensorboard:
        writer.close()
    if save_results:
        results_file.close()

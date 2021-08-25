import core.mcts_widening as mctswidening
import core.mcts as mctsnotwidening

class Trainer():
    """
    Trainer class. Used for a given environment to perform training and validation steps.
    """
    def __init__(self, environment, policy, replay_buffer, curriculum_scheduler, mcts_train_params,
                 mcts_test_params, num_validation_episodes, num_episodes_per_task, batch_size, num_updates_per_episode,
                 verbose=True, autograd_from=-1, autograd_to=100, widening=False):

        self.env = environment
        self.policy = policy
        self.buffer = replay_buffer
        self.curriculum_scheduler = curriculum_scheduler
        self.mcts_train_params = mcts_train_params
        self.mcts_test_params = mcts_test_params

        self.num_validation_episodes = num_validation_episodes
        self.num_episodes_per_task = num_episodes_per_task
        self.batch_size = batch_size
        self.num_updates_per_episode = num_updates_per_episode

        self.verbose = verbose

        self.autograd_from = autograd_from
        self.autograd_to = autograd_to

        self.widening = widening

        # Generate an empty list for each program
        self.iterations_for_each_program = {}
        for p in self.env.programs_library:
            self.iterations_for_each_program[p] = 0


    def perform_validation_step(self, task_index):
        """
        Perform validation steps for the task from index task_index.

        Args:
            task_index: task index

        Returns:
            (rewards, traces lengths)

        """
        validation_rewards = []
        traces_lengths = []
        for _ in range(self.num_validation_episodes):
            # Start new episode
            if self.widening:
                mcts = mctswidening.MCTS(self.policy, self.env, task_index, **self.mcts_test_params)
            else:
                mcts = mctsnotwidening.MCTS(self.policy, self.env, task_index, **self.mcts_test_params)

            # Sample an execution trace with mcts using policy as a prior
            trace = mcts.sample_execution_trace()
            task_reward, trace_length, progs_failed_indices = trace[7], len(trace[3]), trace[10]

            validation_rewards.append(task_reward)
            traces_lengths.append(trace_length)
        return validation_rewards, traces_lengths, progs_failed_indices

    def play_iteration(self, task_index, verbose=False, current_iteration=0):
        """
        Play one training iteration, i.e. select a task, play episodes, store experience in buffer and sample batches
        to perform gradient descent on policy weights.

        """

        # Keep all the losses
        actor_losses = 0
        critic_losses = 0
        arguments_losses = 0

        # Compute the total nodes
        total_nodes = {}
        total_nodes_selected = []

        # Get new task to attempt
        task_name = self.env.get_program_from_index(task_index)
        if self.verbose:
            print('Attempt task {} for {} episodes'.format(task_name, self.num_episodes_per_task))

        # Increment the counter for this program
        self.iterations_for_each_program[task_name] += 1

        check_autograd = False
        if self.autograd_from >= 0:
            if self.autograd_from <= current_iteration <= self.autograd_from+self.autograd_to:
                check_autograd = True

        # Start training on the task
        for episode in range(self.num_episodes_per_task):

            # Start new episode
            if self.widening:
                mcts = mctswidening.MCTS(self.policy, self.env, task_index, **self.mcts_train_params)
            else:
                mcts = mctsnotwidening.MCTS(self.policy, self.env, task_index, **self.mcts_train_params)

            # Sample an execution trace with mcts using policy as a prior
            res = mcts.sample_execution_trace()
            observations, prog_indices, previous_actions_indices, policy_labels, lstm_states, _, _, \
                task_reward, clean_sub_execution, rewards, programs_failed_indices, \
                programs_failed_initstates, programs_failed_states_indices, program_args, \
                total_nodes_expanded, total_nodes_selected_episode = res

            total_nodes_selected.append(total_nodes_selected_episode)
            total_nodes = self.merge_counts(total_nodes, total_nodes_expanded)

            # record trace and store it in buffer only if no problem in sub-programs execution
            if clean_sub_execution:
                # Generates trace
                trace = list(zip(observations, prog_indices, lstm_states, policy_labels, rewards, program_args))
                # Append trace to buffer
                self.buffer.append_trace(trace)
            else:
                if self.verbose:
                    print("Trace has not been stored in buffer.")

                # Decrease statistics of programs that failed
                #for idx in programs_failed_indices:
                    #self.curriculum_scheduler.update_statistics(idx, torch.FloatTensor([0.0]))


            # Train policy on batch
            if self.buffer.get_memory_length() > self.batch_size:
                for _ in range(self.num_updates_per_episode):
                    batch = self.buffer.sample_batch(self.batch_size)
                    if batch is not None:
                        actor_loss, critic_loss, arg_loss, _ = self.policy.train_on_batch(batch, check_autograd)
                        actor_losses += actor_loss
                        critic_losses += critic_loss
                        arguments_losses += arg_loss

            if self.verbose:
                print("Done episode {}/{}".format(episode + 1, self.num_episodes_per_task))

        # Sum up all the exploration results
        for k in total_nodes.keys():
            total_nodes[k] = sum(total_nodes[k])

        return actor_losses/self.num_episodes_per_task, critic_losses/self.num_episodes_per_task,\
               arguments_losses/self.num_episodes_per_task, total_nodes, total_nodes_selected

    def perform_validation(self):
        """
        Perform validation for all the tasks and update curriculum scheduelr statistics.
        """
        if self.verbose:
            print("Start validation .....")
        for idx in self.curriculum_scheduler.get_tasks_of_maximum_level():
            # Evaluate performance on task idx
            v_rewards, v_lengths, programs_failed_indices = self.perform_validation_step(idx)
            # Update curriculum statistics
            self.curriculum_scheduler.update_statistics(idx, v_rewards)

            # Decrease statistics of programs that failed
            #for idx_ in programs_failed_indices:
                #self.curriculum_scheduler.update_statistics(idx_, torch.FloatTensor([0.0]))

    def merge_counts(self, a, b):
        tmp = a.copy()
        for k in b.keys():
            if k in tmp:
                tmp[k] += b[k].copy()
            else:
                tmp[k] = b[k].copy()
        return tmp
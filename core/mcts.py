# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

import numpy as np
import torch

import copy


class MCTS:
    """This class is used to perform a search over the state space for different paths by building
    a tree of visited states. Then this tree is used to get an estimation distribution of
    utility over actions.

    Args:
      policy: Policy to be used as a prior over actions given an state.
      c_puct: Constant that modifies the exploration-exploitation tradeoff of the MCTS algorithm.
      env: The environment considered.
      task_index: The index of the task (index of the corresponding program) we are trying to solve.
      number_of_simulations: The number of nodes that we will be visiting when building an MCTS tree.
      temperature: Another parameter that balances exploration-exploitation in MCTS by adding noise to the priors output by the search.
      max_depth_dict: Dictionary that maps a program level to the allowed number of actions to execute a program of that level
      use_dirichlet_noise: Boolean authorizes or not addition of dirichlet noise to prior during simulations to encourage exploration
      dir_epsilon: Proportion of the original prior distribution kept in the newly-updated prior distribution with dirichlet noise
      dir_noise: Parameter of the Dirichlet distribution
      exploit: Boolean if True leads to sampling from the mcts visit policy instead of taking the argmax
      gamma: discount factor, reward discounting increases with depth of trace
      save_sub_trees: Boolean to save in a node the sub-execution trace of a non-zero program
      recursion_depth: Recursion level of the calling tree
      max_recursion_depth: Max recursion level allowed
      qvalue_temperature: Induces tradeoff between mean qvalue and max qvalue when estimating Q in PUCT criterion
      recursive_penalty: Penalty applied to discounted reward if recursive program does not call itself
    """

    def __init__(self, policy, env, task_index, level_closeness_coeff=1.0,
                 c_puct=1.0, number_of_simulations=100, max_depth_dict={1: 5, 2: 50, 3: 150},
                 temperature=1.0, use_dirichlet_noise=False,
                 dir_epsilon=0.25, dir_noise=0.03, exploit=False, gamma=0.97, save_sub_trees=False,
                 recursion_depth=0, max_recursion_depth=500, max_recursion_program_call=15, qvalue_temperature=1.0,
                 recursive_penalty=0.9, structural_penalty_factor=3, use_structural_constraint = False,
                 penalize_level_0=True, level_0_penalty=1, verbose=True, recursive_total_calls=0,
                 recursive_program_total_calls=0, arguments_correctness_coeff=1.0, args_exploration_prob=0.3,
                 use_arguments=True, normalize_policy=False, max_exploration_nodes=1.0, use_gpu=False):

        self.policy = policy
        self.c_puct = c_puct
        self.level_closeness_coeff = level_closeness_coeff
        self.env = env
        self.task_index = task_index
        self.task_name = env.get_program_from_index(task_index)
        self.recursive_task = env.programs_library[self.task_name]['recursive']
        self.recursive_penalty = recursive_penalty
        self.number_of_simulations = number_of_simulations
        self.temperature = temperature
        self.max_depth_dict = max_depth_dict
        self.dirichlet_noise = use_dirichlet_noise
        self.dir_epsilon = dir_epsilon
        self.dir_noise = dir_noise
        self.exploit = exploit
        self.gamma = gamma
        self.save_sub_trees = save_sub_trees
        self.recursion_depth = recursion_depth
        self.max_recursion_depth = max_recursion_depth
        self.qvalue_temperature = qvalue_temperature
        self.structural_penalty_factor = structural_penalty_factor
        self.use_structural_constraint = use_structural_constraint
        self.penalize_level_0 = penalize_level_0
        self.level_0_penalty = level_0_penalty
        self.verbose = verbose
        self.recursive_total_calls = recursive_total_calls
        self.recursive_program_total_calls = recursive_program_total_calls
        self.max_recursion_program_call = max_recursion_program_call
        self.arguments_correctness_coeff = arguments_correctness_coeff
        self.args_exploration_prob = args_exploration_prob
        self.use_arguments = use_arguments
        self.normalize_policy = normalize_policy
        self.max_exploration_nodes = max_exploration_nodes

        self.total_node_expanded = None

        if not self.use_arguments:
            self.STOP_action_name = "STOP_[0, 0, 0]"
        else:
            self.STOP_action_name = "STOP"

        assert self.level_0_penalty >= 0, "Level 0 custom penalty must be a positive number!"

        # record if all sub-programs executed correctly (useful only for programs of level > 1)
        self.clean_sub_executions = True

        # recursive trees parameters
        self.sub_tree_params = {'number_of_simulations': 5, 'max_depth_dict': self.max_depth_dict,
            'temperature': self.temperature, 'c_puct': self.c_puct, 'exploit': True,
            'level_closeness_coeff': self.level_closeness_coeff, 'gamma': self.gamma,
            'save_sub_trees': self.save_sub_trees, 'recursion_depth': recursion_depth+1,
            'max_recursion_depth': self.max_recursion_depth, 'use_structural_constraint': self.use_structural_constraint,
            'penalize_level_0': self.penalize_level_0, 'level_0_penalty': self.level_0_penalty,
            'verbose': self.verbose, 'recursive_total_calls': self.recursive_total_calls,
            'recursive_program_total_calls': self.recursive_program_total_calls,
            'max_recursion_program_call': self.max_recursion_program_call,
            'use_arguments': self.use_arguments}


    def _expand_node(self, node):
        """Used for previously unvisited nodes. It evaluates each of the possible child and
        initializes them with a score derived from the prior output by the policy network.

        Args:
          node: Node to be expanded

        Returns:
          node now expanded, value, hidden_state, cell_state

        """
        program_index, observation, env_state, h, c, depth, program_call_count, previously_called_index = (
            node["program_index"],
            node["observation"],
            node["env_state"],
            node["h_lstm"],
            node["c_lstm"],
            node["depth"],
            node["program_call_count"],
            node["program_from_parent_index"]
        )

        with torch.no_grad():
            mask = self.env.get_mask_over_actions(program_index)
            priors, value, new_args, new_h, new_c = self.policy.forward_once(observation, program_index, h, c)

            if not self.use_arguments:
                new_args = torch.FloatTensor(np.ones(len(self.env.arguments)))

            # mask actions
            priors = priors * torch.FloatTensor(mask)
            priors = torch.squeeze(priors)
            priors = priors.cpu().numpy()

            if self.dirichlet_noise:
                priors = (1 - self.dir_epsilon) * priors + self.dir_epsilon * np.random.dirichlet([self.dir_noise] * priors.size)

            policy_indexes = [prog_idx for prog_idx, x in enumerate(mask) if x == 1]
            policy_probability = [priors[prog_idx] for prog_idx, x in enumerate(mask) if x == 1]

            if self.normalize_policy:
                policy_probability = self.fix_policy(torch.FloatTensor(policy_probability)).numpy()

        # Current new nodes
        new_nodes = []

        # Initialize its children with its probability of being chosen
        for prog_index, prog_proba in zip(policy_indexes, policy_probability):

            # mask arguments
            mask_args = self.env.get_mask_over_args(prog_index)
            new_args_masked = new_args * torch.FloatTensor(mask_args)
            new_args_masked = torch.squeeze(new_args_masked)
            new_args_masked = new_args_masked.cpu().numpy()

            if self.dirichlet_noise and self.use_arguments:
                new_args_masked = (1 - self.dir_epsilon) * new_args_masked \
                         + self.dir_epsilon * np.random.dirichlet([self.dir_noise] * new_args_masked.size)

            if self.use_arguments:

                args_probability = [new_args_masked[arg_idx] for arg_idx, y in enumerate(mask_args) if y == 1]
                args_indexs = [arg_idx for arg_idx, y in enumerate(mask_args) if y == 1]

                # Normalize the arguments probability such to ensure better numerical
                # stability
                if self.normalize_policy:
                    args_probability = self.fix_policy(torch.FloatTensor(args_probability)).numpy()

                for arg_index, args_proba in zip(args_indexs, args_probability):

                    new_child = {
                        "parent": node,
                        "childs": [],
                        "visit_count": 0.0,
                        "total_action_value": [],
                        "prior": float(prog_proba*args_proba),
                        "program_from_parent_index": prog_index,
                        "program_index": program_index,
                        "observation": observation,
                        "env_state": env_state,
                        "h_lstm": new_h.clone(),
                        "c_lstm": new_c.clone(),
                        "args": self.env.arguments[arg_index],
                        "args_index": arg_index,
                        "selected": False,
                        "depth": depth + 1,
                        "program_call_count": program_call_count.copy()
                    }

                    # If we are calling again the same program, then we increment
                    # the program counter for that program. This is done to record
                    # eventual while presents in the code.
                    if prog_index == previously_called_index:
                        new_child["program_call_count"][prog_index] +=1

                    # Add the new node in a temporary array
                    new_nodes.append(new_child)

            else:

                    argument_det = self.env.programs_library[self.env.get_program_from_index(prog_index)]["determ_args"]

                    new_child = {
                        "parent": node,
                        "childs": [],
                        "visit_count": 0.0,
                        "total_action_value": [],
                        "prior": float(prog_proba),
                        "program_from_parent_index": prog_index,
                        "program_index": program_index,
                        "observation": observation,
                        "env_state": env_state,
                        "h_lstm": new_h.clone(),
                        "c_lstm": new_c.clone(),
                        "args": argument_det,
                        "args_index": 0,
                        "selected": False,
                        "depth": depth + 1,
                        "program_call_count": program_call_count.copy()
                    }

                    # If we are calling again the same program, then we increment
                    # the program counter for that program. This is done to record
                    # eventual while presents in the code.
                    if prog_index == previously_called_index:
                        new_child["program_call_count"][prog_index] += 1

                    new_nodes.append(new_child)

        # Append the new nodes to graph
        node["childs"] = new_nodes

        # This reward will be propagated backwards through the tree
        value = float(value)
        return node, value, new_h.clone(), new_c.clone(), len(new_nodes)

    def _compute_q_value(self, node):
        if node["visit_count"] > 0.0:
            values = torch.FloatTensor(node['total_action_value'])
            softmax = torch.exp(self.qvalue_temperature * values)
            softmax = softmax / softmax.sum()
            q_val_action = float(torch.dot(softmax, values))
        else:
            q_val_action = 0.0
        return q_val_action

    def _estimate_q_val(self, node):
        """Estimates the Q value over possible actions in a given node, and returns the action
        and the child that have the best estimated value.

        Args:
          node: Node to evaluate its possible actions.

        Returns:
          best child found from this node.

        """
        best_child = None
        best_val = -np.inf
        # Iterate all the children to fill up the node dict and estimate Q val.
        # Then track the best child found according to the Q value estimation
        for child in node["childs"]:
            if child["prior"] > 0.0:
                q_val_action = self._compute_q_value(child)

                action_utility = (self.c_puct * child["prior"] * np.sqrt(node["visit_count"])
                                  * (1.0 / (1.0 + child["visit_count"])))
                q_val_action += action_utility
                parent_prog_lvl = self.env.programs_library[self.env.idx_to_prog[node['program_index']]]['level']
                action_prog_lvl = self.env.programs_library[self.env.idx_to_prog[child['program_from_parent_index']]]['level']

                #total_arguments = self.env.programs_library[self.env.idx_to_prog[child['program_from_parent_index']]]['args']
                #current_arguments = sum(child["args"])
                #
                # Add penalty for the actions
                #if not (current_arguments in total_arguments):
                #    arguments_penalty = self.arguments_correctness_coeff*np.exp(-4)
                #else:
                #    arguments_penalty = self.arguments_correctness_coeff*np.exp(-1)
                #
                #q_val_action += arguments_penalty

                if parent_prog_lvl == action_prog_lvl:
                    # special treatment for calling the same program or a level 0 action.
                    action_level_closeness = self.level_closeness_coeff * np.exp(-1)
                elif action_prog_lvl == 0 and not self.penalize_level_0:
                    action_level_closeness = self.level_closeness_coeff * np.exp(-self.level_0_penalty)
                elif action_prog_lvl > 0 or (action_prog_lvl == 0 and self.penalize_level_0):
                    action_level_closeness = self.level_closeness_coeff * np.exp(-(parent_prog_lvl - action_prog_lvl))
                else:
                    # special treatment for STOP action
                    action_level_closeness = self.level_closeness_coeff * np.exp(-1)

                q_val_action += action_level_closeness

                if self.use_structural_constraint:
                    structural_penalty_value = self.return_structural_penalty(child,
                                                                 self.env.prog_to_structural_condition[
                                                                     self.env.get_program_from_index(child["program_index"])
                                                                 ]
                                                                 )
                    q_val_action += structural_penalty_value

                if q_val_action > best_val:
                    best_val = q_val_action
                    best_child = child

        return best_child

    def _sample_policy(self, root_node):
        """Sample an action from the policies and q_value distributions that were previously sampled.

        Args:
          root_node: Node to choose the best action from. It should be the root node of the tree.

        Returns:
          Tuple containing the sampled action and the probability distribution build normalizing visits_policy.
        """
        visits_policy = []
        for child in root_node["childs"]:
            if child["prior"] > 0.0:
                visits_policy.append([child['program_from_parent_index'], child["visit_count"], child["args_index"]])

        mcts_policy = torch.zeros(1, self.env.get_num_programs())
        args_policy = torch.zeros(1, len(self.env.arguments))

        for prog_index, visit, arg_index in visits_policy:
            mcts_policy[0, prog_index] += visit

            if self.use_arguments:
                args_policy[0, arg_index] += visit

        if self.exploit:
            mcts_policy = mcts_policy / mcts_policy.sum()

            if self.use_arguments:
                args_policy = args_policy / args_policy.sum()
            else:
                args_policy = torch.FloatTensor([1])

            #mcts_policy = self.fix_policy(mcts_policy)
            #args_policy = self.fix_policy(args_policy)

            return mcts_policy, args_policy, int(torch.argmax(mcts_policy)), int(torch.argmax(args_policy))
        else:
            mcts_policy = torch.pow(mcts_policy, self.temperature)
            mcts_policy = mcts_policy / mcts_policy.sum()

            if mcts_policy.sum() == 0.0:
                mcts_policy = torch.ones(1, self.env.get_num_programs())/self.env.get_num_programs()

            if self.use_arguments:
                args_policy = torch.pow(args_policy, self.temperature)
                args_policy = args_policy / args_policy.sum()

                if args_policy.sum() == 0.0:
                    args_policy = torch.ones(1, len(self.env.arguments)) / len(self.env.arguments)
            else:
                args_policy = torch.FloatTensor([1])

            #mcts_policy = self.fix_policy(mcts_policy)
            #args_policy = self.fix_policy(args_policy)

            if self.use_arguments:
                args_sampled = int(torch.multinomial(args_policy, 1)[0, 0])
            else:
                args_sampled = 0

            return mcts_policy, args_policy, int(torch.multinomial(mcts_policy, 1)[0, 0]), args_sampled

    def _run_simulation(self, node):
        """Run one simulation in tree. This function is recursive.

        Args:
          node: root node to run the simulation from
          program_index: index of the current calling program

        Returns:
            (if the max depth has been reached or not, if a node has been expanded or not, node reached at the end of the simulation)

        """

        stop = False
        max_depth_reached = False
        max_recursion_reached = False
        has_expanded_a_node = False
        failed_simulation = False
        value = None
        program_level = self.env.get_program_level_from_index(node['program_index'])

        total_node_expanded = 0
        total_sub_selected_nodes = 0

        while not stop and not max_depth_reached and not has_expanded_a_node and self.clean_sub_executions and not max_recursion_reached:

            if node['depth'] >= self.max_depth_dict[program_level]:
                max_depth_reached = True

            elif len(node['childs']) == 0:
                _, value, state_h, state_c, total_node_added = self._expand_node(node)

                total_node_expanded += total_node_added

                has_expanded_a_node = True

                if total_node_added == 0:
                    failed_simulation = True
                    break

            else:
                best_node = self._estimate_q_val(node)

                # Check this corner case. If this happened, then we
                # failed this simulation and its reward will be -1.
                if best_node is None:
                    failed_simulation = True
                    break
                else:
                    node = best_node

                program_to_call_index = node['program_from_parent_index']
                program_to_call = self.env.get_program_from_index(program_to_call_index)
                arguments = node['args']

                if program_to_call_index == self.env.programs_library[self.STOP_action_name]['index']:
                    stop = True

                elif self.env.programs_library[program_to_call]['level'] == 0:
                    observation = self.env.act(program_to_call, arguments)
                    node['observation'] = observation
                    node['env_state'] = self.env.get_state()

                else:
                    # check if call corresponds to a recursive call
                    if program_to_call_index == self.task_index:
                        self.recursive_call = True
                    # if never been done, compute new tree to execute program
                    if node['visit_count'] == 0.0:

                        # If we are recursive, then we compute the max recursion depth
                        # by taking into account also the tree depth itself
                        if self.recursive_total_calls >= self.max_recursion_depth:
                            max_recursion_reached = True
                            if self.verbose:
                                print("Reached the maximum_recursion_depth {}, with recursion depth {} ".format(self.recursive_total_calls, self.recursion_depth))
                            continue

                        # If we reached the maximum amount of call for the given subprogram then return
                        if self.recursive_program_total_calls >= self.max_recursion_program_call:
                            if self.verbose:
                                print("Reached the maximum_recursion_depth for program {}, with recursion depth {} "
                                      .format(self.env.get_program_from_index(self.task_index),
                                              self.max_recursion_program_call))
                            continue

                        sub_mcts_init_state = self.env.get_state()

                        # Copy sub_tree_params and increase node counts
                        copy_ = copy.deepcopy(self.sub_tree_params)

                        # Increase the total depth of the tree
                        copy_["recursive_total_calls"] += node["depth"]

                        # Increase the recursion depth for the given program
                        if self.recursive_call:
                            if self.verbose:
                                print("Increased recursive program count.")
                            copy_["recursive_program_total_calls"] += 1

                        sub_mcts = MCTS(self.policy, self.env, program_to_call_index, **copy_)
                        sub_trace = sub_mcts.sample_execution_trace()
                        sub_task_reward, sub_root_node, sub_total_nodes, sub_selected_nodes = sub_trace[7], sub_trace[6], sub_trace[14], sub_trace[15]

                        total_sub_selected_nodes += sub_selected_nodes

                        # Add submcts node expanded
                        for k in sub_total_nodes:
                            total_node_expanded += sum(sub_total_nodes[k])

                        # if save sub tree is true, then store sub root node
                        if self.save_sub_trees:
                            node['sub_root_node'] = sub_root_node
                        # allows tree saving of first non zero program encountered

                        # check that sub tree executed correctly
                        self.clean_sub_executions &= (sub_task_reward > -1.0)
                        if not self.clean_sub_executions:
                            if self.verbose:
                                print('program {} did not execute correctly'.format(program_to_call))
                            self.programs_failed_indices.append(program_to_call_index)
                            self.programs_failed_indices += sub_mcts.programs_failed_indices
                            self.programs_failed_initstates.append(sub_mcts_init_state)

                        observation = self.env.get_observation()
                    else:
                        self.env.reset_to_state(node['env_state'])
                        observation = self.env.get_observation()

                    node['observation'] = observation
                    node['env_state'] = self.env.get_state()

        return max_depth_reached, has_expanded_a_node, node, value, failed_simulation, total_node_expanded, total_sub_selected_nodes

    def _play_episode(self, root_node):
        """Performs an MCTS search using the policy network as a prior and returns a sequence of improved decisions.

        Args:
          root_node: Root node of the tree.

        Returns:
            (Final node reached at the end of the episode, boolean stating if the max depth allowed has been reached).

        """
        stop = False
        max_depth_reached = False
        illegal_action = False

        selected_nodes_count = 0

        while not stop and not max_depth_reached and not illegal_action and self.clean_sub_executions:

            program_level = self.env.get_program_level_from_index(root_node['program_index'])
            # tag node as from the final execution trace (for visualization purpose)
            root_node["selected"] = True

            if root_node['depth'] >= self.max_depth_dict[program_level]:
                max_depth_reached = True

            else:
                env_state = root_node["env_state"]

                # record obs, progs and lstm states only if they correspond to the current task at hand
                self.lstm_states.append((root_node['h_lstm'], root_node['c_lstm']))
                self.programs_index.append(root_node['program_index'])
                self.observations.append(root_node['observation'])
                self.previous_actions.append(root_node['program_from_parent_index'])
                self.program_arguments.append(root_node['args'])
                self.rewards.append(None)

                total_node_expanded_simulation = 0

                # Spend some time expanding the tree from your current root node
                for _ in range(self.number_of_simulations):
                    # run a simulation
                    self.recursive_call = False
                    simulation_max_depth_reached, has_expanded_node, node, value, failed_simulation, total_node_expanded, total_sub_selected_nodes = self._run_simulation(root_node)
                    total_node_expanded_simulation += total_node_expanded
                    selected_nodes_count += total_sub_selected_nodes

                    # get reward
                    if failed_simulation:
                        value = -1.0
                    elif not simulation_max_depth_reached and not has_expanded_node:
                        # if node corresponds to end of an episode, backprogagate real reward
                        reward = self.env.get_reward()
                        if reward > 0:
                            value = self.env.get_reward() * (self.gamma ** node['depth'])
                            if self.recursive_task and not self.recursive_call:
                                # if recursive task but do not called itself, add penalization
                                value -= self.recursive_penalty
                        else:
                            value = -1.0

                    elif simulation_max_depth_reached:
                        # if episode stops because the max depth allowed was reached, then reward = -1
                        value = -1.0

                    value = float(value)

                    # Propagate information backwards
                    while node["parent"] is not None:
                        node["visit_count"] += 1
                        node["total_action_value"].append(value)
                        node = node["parent"]
                    # Root node is not included in the while loop
                    self.root_node["total_action_value"].append(value)
                    self.root_node["visit_count"] += 1

                    # Go back to current env state
                    self.env.reset_to_state(env_state)

                # Record how many nodes we expanded for this simulation
                self.total_node_expanded[self.task_index].append(total_node_expanded_simulation)

                # Sample next action
                mcts_policy, args_policy, program_to_call_index, args_to_call_index = self._sample_policy(root_node)
                if program_to_call_index == self.task_index:
                    self.global_recursive_call = True

                # Set new root node
                if self.use_arguments:
                    root_node = [child for child in root_node["childs"]
                                 if child["program_from_parent_index"] == program_to_call_index
                                 and child["args_index"] == args_to_call_index]
                else:
                    root_node = [child for child in root_node["childs"]
                                 if child["program_from_parent_index"] == program_to_call_index]

                # If we choose an illegal action from this point, we exit
                if len(root_node) == 0:
                    root_node = None
                    illegal_action = True
                else:
                    root_node = root_node[0]

                selected_nodes_count += 1

                # Record mcts policy
                if self.use_arguments:
                    self.mcts_policies.append(torch.cat([mcts_policy, args_policy], dim=1))
                else:
                    self.mcts_policies.append(mcts_policy)

                # Apply chosen action
                if not illegal_action:
                    if program_to_call_index == self.env.programs_library[self.STOP_action_name]['index']:
                        stop = True
                    else:
                        self.env.reset_to_state(root_node["env_state"])

        return root_node, max_depth_reached, illegal_action, selected_nodes_count


    def sample_execution_trace(self):
        """
        Args:
          init_observation: initial observation before playing an episode

        Returns:
            (a sequence of (e_t, i_t), a sequence of probabilities over programs, a sequence of (h_t, c_t), if the maximum depth allowed has been reached)
        """

        # start the task
        init_observation, env_index, env_total_size = self.env.start_task(self.task_index)
        with torch.no_grad():
            state_h, state_c = self.policy.init_tensors()
            self.env_init_state = self.env.get_state()

            self.root_node = {
                "parent": None,
                "childs": [],
                "visit_count": 1,
                "total_action_value": [],
                "prior": None,
                "program_index": self.task_index,
                "program_from_parent_index": None,
                "observation": init_observation,
                "env_state": self.env_init_state,
                "h_lstm": state_h.clone(),
                "c_lstm": state_c.clone(),
                "args": np.array([0,0,0]),
                "args_original": np.array([0, 0, 0]),
                "args_index": 0,
                "depth": 0,
                "selected": True,
                "program_call_count": [0 for _ in range(0, len(self.env.programs_library))]
            }

            # prepare empty lists to store trajectory
            self.programs_index = []
            self.observations = []
            self.previous_actions = []
            self.mcts_policies = []
            self.lstm_states = []
            self.program_arguments = []
            self.rewards = []
            self.programs_failed_indices = []
            self.programs_failed_initstates = []
            self.programs_failed_states_indices = [[] for i in range(len(self.env.programs_library))]

            self.global_recursive_call = False

            self.total_node_expanded = {self.task_index: []}

            # play an episode
            final_node, max_depth_reached, illegal_action, selected_nodes_count = self._play_episode(self.root_node)

            if not illegal_action:
                final_node['selected'] = True

        # compute final task reward (with gamma penalization)
        reward = self.env.get_reward()
        if reward > 0 and not illegal_action:
            task_reward = reward * (self.gamma**final_node['depth'])
            if self.recursive_task and not self.global_recursive_call:
                # if recursive task but do not called itself, add penalization
                task_reward -= self.recursive_penalty
        else:
            self.programs_failed_states_indices[self.task_index].append((env_index, env_total_size))
            self.env.update_failing_envs(self.env_init_state, self.env.get_program_from_index(self.task_index))
            task_reward = -1

        # Replace None rewards by the true final task reward
        self.rewards = list(map(lambda x: torch.FloatTensor([task_reward]) if x is None else torch.FloatTensor([x]), self.rewards))

        # end task
        self.env.end_task()

        return self.observations, self.programs_index, self.previous_actions, self.mcts_policies, \
               self.lstm_states, max_depth_reached, self.root_node, task_reward, self.clean_sub_executions, self.rewards, \
               self.programs_failed_indices, self.programs_failed_initstates, self.programs_failed_states_indices, \
               self.program_arguments, self.total_node_expanded, selected_nodes_count

    def return_structural_penalty(self, node, condition=None):
        """
        Return structural penalty given a MCTS node, the program level and a structural
        condition (like WHILE, IF, SEQUENTIAL, etc.)

        :param node: the MCTS node we are evaluating
        :param program_level: the level of the program we are computing
        :param condition: the structural constraint
        :return:
        """

        if condition == "WHILE":
            max_depth = self.max_depth_dict[node["program_index"]]
            total_called = sum(node["program_call_count"])
            penalty = self.structural_penalty_factor * np.exp(-(max_depth-total_called))
        elif condition == "SEQUENTIAL":
            # Since the execution is sequential. We want to penalize
            # calling the same instruction twice.
            penalty = self.structural_penalty_factor * np.exp(-sum(node["program_call_count"]))
        else:
            penalty = 0
        return penalty

    def fix_policy(self, policy):
        """
        This fix potential issues which happens withing the policy. Namely, NaN probabilities
        or negative probabilites (which should not happen anyway).
        :param policy: the policy we need to fix
        :return: a safe version of the policy
        """

        safe_policy = torch.where(torch.isnan(policy), torch.zeros_like(policy), policy)
        safe_policy = torch.max(safe_policy, torch.tensor([0.]))

        # This means that we changed something. Therefore we need to recompute
        # the policy taking into account the corrected values
        if safe_policy.sum() != 1.0 and safe_policy.sum() != 0.0:
            safe_policy = safe_policy / safe_policy.sum()

        return safe_policy

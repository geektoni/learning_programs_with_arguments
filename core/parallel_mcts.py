import numpy as np
import torch

from .mcts_widening import MCTS

from itertools import repeat

def _run_simulation(node, env, mcts):
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
    program_level = env.get_program_level_from_index(node['program_index'])

    total_node_expanded = 0
    total_sub_selected_nodes = 0

    while not stop and not max_depth_reached and not has_expanded_a_node and clean_sub_executions and not max_recursion_reached:

        if node['depth'] >= mcts.max_depth_dict[program_level]:
            max_depth_reached = True

        elif len(node['childs']) == 0:
            _, value, state_h, state_c, total_node_added = mcts._expand_node(node, 100)
            has_expanded_a_node = False

            total_node_expanded += total_node_added

            if total_node_added == 0:
                failed_simulation = True
                break
        else:

            # If we reach the correct point, then we add new nodes to the current node
            # before choosing the best action. Moreover, if we are exploiting, then we do
            # not need to explore more nodes, since the one we will sample will mostly be
            # the correct ones.
            # if log_child_increment \
            #    and not node["expanded"] and not self.exploit:
            #    _, value, state_h, state_c = self._expand_node(node, self.default_childs)

            best_node = self._estimate_q_val(node)

            # Check this corner case. If this happened, then we
            # failed this simulation and its reward will be -1.
            if best_node is None:
                failed_simulation = True
                break
            else:
                node = best_node

            program_to_call_index = node['program_from_parent_index']
            program_to_call = env.get_program_from_index(program_to_call_index)
            arguments = node['args']

            if program_to_call_index == env.programs_library[self.STOP_action_name]['index']:
                stop = True

            elif env.programs_library[program_to_call]['level'] == 0:
                observation = env.act(program_to_call, arguments)
                node['observation'] = observation
                node['env_state'] = env.get_state()

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
                            print("Reached the maximum_recursion_depth {}, with recursion depth {} ".format(
                                self.recursive_total_calls, self.recursion_depth))
                        continue

                    # If we reached the maximum amount of call for the given subprogram then return
                    if self.recursive_program_total_calls >= self.max_recursion_program_call:
                        if self.verbose:
                            print("Reached the maximum_recursion_depth for program {}, with recursion depth {} "
                                  .format(env.get_program_from_index(self.task_index),
                                          self.max_recursion_program_call))
                        continue

                    sub_mcts_init_state = env.get_state()

                    # Copy sub_tree_params and increase node counts
                    copy_ = copy.deepcopy(self.sub_tree_params)

                    # Increase the total depth of the tree
                    copy_["recursive_total_calls"] += node["depth"]

                    # Increase the recursion depth for the given program
                    if self.recursive_call:
                        if self.verbose:
                            print("Increased recursive program count.")
                        copy_["recursive_program_total_calls"] += 1

                    sub_mcts = MCTS(self.policy, env, program_to_call_index, **copy_)
                    sub_trace = sub_mcts.sample_execution_trace()
                    sub_task_reward, sub_root_node, sub_total_nodes, sub_selected_nodes = sub_trace[7], sub_trace[6], \
                                                                                          sub_trace[14], sub_trace[15]

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

                    observation = env.get_observation()
                else:
                    env.reset_to_state(node['env_state'])
                    observation = env.get_observation()

                node['observation'] = observation
                node['env_state'] = env.get_state()

    return max_depth_reached, has_expanded_a_node, node, value, failed_simulation, total_node_expanded, total_sub_selected_nodes
import torch


class NetworkOnly:

    def __init__(self, policy, env, max_depth_dict, use_arguments):
        self.policy = policy
        self.env = env

        self.use_arguments = use_arguments

        if not self.use_arguments:
            self.STOP_action_name = "STOP_[0, 0, 0]"
        else:
            self.STOP_action_name = "STOP"

        self.stop_index = env.programs_library[self.STOP_action_name]['index']
        self.max_depth_dict = max_depth_dict
        self.clean_sub_executions = True

    def play(self, task_index):
        programs_called = []
        # Get max depth for this program
        task_level = self.env.get_program_level_from_index(task_index)
        max_depth = self.max_depth_dict[task_level]
        depth = 0

        # Start new task and initialize LSTM
        observation, _, _ = self.env.start_task(task_index)
        h, c = self.policy.init_tensors()

        while self.clean_sub_executions and depth <= max_depth:
            # Compute mask
            mask = self.env.get_mask_over_actions(task_index)
            # Compute priors
            priors, value, arguments, h, c = self.policy.forward_once(observation, task_index, h, c)
            # Mask actions
            priors = priors * torch.FloatTensor(mask)
            priors = torch.squeeze(priors)
            # Choose action according to argmax over priors
            program_index = torch.argmax(priors).item()
            program_name = self.env.get_program_from_index(program_index)
            programs_called.append(program_name)

            # Mask arguments and choose arguments
            arguments_mask = self.env.get_mask_over_args(program_index)
            arguments = arguments * torch.FloatTensor(arguments_mask)
            arguments_index = torch.argmax(arguments).item()
            arguments_list = self.env.arguments[arguments_index]

            depth += 1

            # Apply action
            if program_name == self.STOP_action_name:
                break
            elif self.env.programs_library[program_name]['level'] == 0:
                observation = self.env.act(program_name, arguments_list)
            else:
                r, _ = self.play(task_index=program_index)
                if r < 1.0:
                    self.clean_sub_executions = False
                observation = self.env.get_observation()
        # Get final reward and end task
        if depth <= max_depth:
            reward = self.env.get_reward()
        else:
            reward = 0.0
        self.env.end_task()
        return reward, programs_called

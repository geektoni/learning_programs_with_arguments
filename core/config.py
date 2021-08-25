from collections import OrderedDict

# Contains NPI hyperparameters

# Environment agnostic space dimensions
encoding_dim = 32                           # dimension D of s_t
program_embedding_dim = 256                 # size P of program embedding vector

# LSTM hyper-param
hidden_size = 128                           # size of hidden state h_t

# Optimizer hyper-param
learning_rate = 1e-4                        # learning rate for the policy optimizer

# Curriculum hyper-params
reward_threshold = 0.97                    # reward threshold to increase the tasks levels in curriculum strategy

# MCTS hyper-params
number_of_simulations = 200                 # number of simulations played before taking an action
c_puct = 0.5                                # trade-off exploration/exploitation in mcts
temperature = 1.3                           # coefficient to artificially increase variance in mcts policy distributions
level_closeness_coeff = 3.0                 # importance given to higher level programs
penalize_level_0 = True                     # penalize level 0 action when computing Q-value
level_0_custom_penalty = 1                  # custom penalty for the level 0 action

# Training hyper-params
num_iterations = 1000000                    # total number of iterations, one iteration corresponds to one task
num_episodes_per_task = 20                  # number of episodes played for each new task attempted
batch_size = 256                            # training batch size
buffer_max_length = 2000                    # replay buffer max length
num_updates_per_episode = 2                 # number of gradient descents for every episode played
gamma = 0.97                                # discount factor to penalize long execution traces
proba_replay_buffer = 0.5                   # probability of sampling positive reward experience in buffer

# Validation hyper-params
num_validation_episodes = 25                # number of episodes played for validation
number_of_simulations_for_validation = 5    # number of simulations used in the tree for validation (when exploit = True)

# Constraints
structural_constraint = False               # use structural constraint when learning programs

# Maximum iterations for each program
max_training_iterations = OrderedDict(sorted({'PARTITION_UPDATE': -1,
                                              'PARTITION': -1,
                                              'SAVE_LOAD_PARTITION': -1,
                                              'QUICKSORT_UPDATE_REC': -1,
                                              'QUICKSORT_UPDATE': -1,
                                              'QUICKSORT': -1,
                                              'PARTITION_UPDATE_[0, 0, 0]': -1,
                                              'PARTITION_[0, 0, 0]': -1,
                                              'SAVE_LOAD_PARTITION_[0, 0, 0]': -1,
                                              'QUICKSORT_UPDATE_REC_[0, 0, 0]': -1,
                                              'QUICKSORT_UPDATE_[0, 0, 0]': -1,
                                              'QUICKSORT_[0, 0, 0]': -1,
                                              }.items()))

predefined_hyperparameters = OrderedDict(sorted({'PARTITION_UPDATE': (0.1, 0.5),
                                                 'PARTITION': (-1, -1),
                                                 'SAVE_LOAD_PARTITION': (-1, -1),
                                                 'QUICKSORT_UPDATE_REC': (-1, -1),
                                                 'QUICKSORT_UPDATE': (-1, -1),
                                                 'QUICKSORT': (-1, -1),
                                                 'PARTITION_UPDATE_[0, 0, 0]': (-1, -1),
                                                 'PARTITION_[0, 0, 0]': (-1, -1),
                                                 'SAVE_LOAD_PARTITION_[0, 0, 0]': (-1, -1),
                                                 'QUICKSORT_UPDATE_REC_[0, 0, 0]': (-1, -1),
                                                 'QUICKSORT_UPDATE_[0, 0, 0]': (-1, -1),
                                                 'QUICKSORT_[0, 0, 0]': (-1, -1),
                                                 }.items()))

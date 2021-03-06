import numpy as np

from collections import OrderedDict

programs_library = OrderedDict(sorted({'STOP': {'level': -1, 'recursive': False},
                                                        'PTR_1_LEFT': {'level': 0, 'recursive': False},
                                                        'PTR_2_LEFT': {'level': 0, 'recursive': False},
                                                        'PTR_3_LEFT': {'level': 0, 'recursive': False},
                                                        'PTR_1_RIGHT': {'level': 0, 'recursive': False},
                                                        'PTR_2_RIGHT': {'level': 0, 'recursive': False},
                                                        'PTR_3_RIGHT': {'level': 0, 'recursive': False},
                                                        'SWAP': {'level': 0, 'recursive': False},
                                                        'SWAP_PIVOT': {'level': 0, 'recursive': False},
                                                        'PUSH': {'level': 0, 'recursive': False},
                                                        'POP': {'level': 0, 'recursive': False},
                                                        'SAVE_PTR_1': {'level': 0, 'recursive': False},
                                                        'LOAD_PTR_1': {'level': 0, 'recursive': False},
                                                        'PARTITION_UPDATE': {'level': 1, 'recursive': False},
                                                        'PARTITION': {'level': 2, 'recursive': False},
                                                        'SAVE_LOAD_PARTITION': {'level': 3, 'recursive': False},
                                                        'QUICKSORT_UPDATE': {'level': 4, 'recursive': False},
                                                        'QUICKSORT': {'level': 5, 'recursive': False}}.items()))

programs_library_without_partition_update = OrderedDict(sorted({'STOP': {'level': -1, 'recursive': False, 'args': [0]},
                                                        'PTR_LEFT': {'level': 0, 'recursive': False, 'args': [1,2,3]},
                                                        'PTR_RIGHT': {'level': 0, 'recursive': False, 'args': [1,2,3]},
                                                        'SWAP': {'level': 0, 'recursive': False, 'args': [2]},
                                                        'PUSH': {'level': 0, 'recursive': False, 'args': [0]},
                                                        'POP': {'level': 0, 'recursive': False, 'args': [0]},
                                                        'SAVE_PTR': {'level': 0, 'recursive': False, 'args': [1]},
                                                        'LOAD_PTR': {'level': 0, 'recursive': False, 'args': [1]},
                                                        'PARTITION': {'level': 1, 'recursive': False, 'args': [0]},
                                                        'SAVE_LOAD_PARTITION': {'level': 2, 'recursive': False, 'args': [0]},
                                                        'QUICKSORT_UPDATE': {'level': 3, 'recursive': False, 'args': [0]},
                                                        'QUICKSORT': {'level': 4, 'recursive': False,'args': [0]}}.items()))

programs_library_without_save_load_partition = OrderedDict(sorted({'STOP': {'level': -1, 'recursive': False, 'args': [0]},
                                                        'PTR_LEFT': {'level': 0, 'recursive': False, 'args': [1,2,3]},
                                                        'PTR_RIGHT': {'level': 0, 'recursive': False, 'args': [1,2,3]},
                                                        'SWAP': {'level': 0, 'recursive': False, 'args': [2]},
                                                        'PUSH': {'level': 0, 'recursive': False, 'args': [0]},
                                                        'POP': {'level': 0, 'recursive': False, 'args': [0]},
                                                        'SAVE_PTR': {'level': 0, 'recursive': False, 'args': [1]},
                                                        'LOAD_PTR': {'level': 0, 'recursive': False, 'args': [1]},
                                                        'PARTITION_UPDATE': {'level': 1, 'recursive': False, 'args': [0]},
                                                        'PARTITION': {'level': 2, 'recursive': False, 'args': [0]},
                                                        'QUICKSORT_UPDATE': {'level': 3, 'recursive': False, 'args': [0]},
                                                        'QUICKSORT': {'level': 4, 'recursive': False,'args': [0]}}.items()))

programs_library_reduced = OrderedDict(sorted({'STOP': {'level': -1, 'recursive': False, 'args': [0]},
                                                        'PTR_LEFT': {'level': 0, 'recursive': False, 'args': [1,2,3]},
                                                        'PTR_RIGHT': {'level': 0, 'recursive': False, 'args': [1,2,3]},
                                                        'SWAP': {'level': 0, 'recursive': False, 'args': [2]},
                                                        'PUSH': {'level': 0, 'recursive': False, 'args': [0]},
                                                        'POP': {'level': 0, 'recursive': False, 'args': [0]},
                                                        'SAVE_PTR': {'level': 0, 'recursive': False, 'args': [1]},
                                                        'LOAD_PTR': {'level': 0, 'recursive': False, 'args': [1]},
                                                        'PARTITION': {'level': 1, 'recursive': False, 'args': [0]},
                                                        'QUICKSORT_UPDATE': {'level': 2, 'recursive': False, 'args': [0]},
                                                        'QUICKSORT': {'level': 3, 'recursive': False,'args': [0]}}.items()))

programs_library_recursive_quicksort_update = OrderedDict(sorted({'STOP': {'level': -1, 'recursive': False},
                                                        'PTR_1_LEFT': {'level': 0, 'recursive': False},
                                                        'PTR_2_LEFT': {'level': 0, 'recursive': False},
                                                        'PTR_3_LEFT': {'level': 0, 'recursive': False},
                                                        'PTR_1_RIGHT': {'level': 0, 'recursive': False},
                                                        'PTR_2_RIGHT': {'level': 0, 'recursive': False},
                                                        'PTR_3_RIGHT': {'level': 0, 'recursive': False},
                                                        'SWAP': {'level': 0, 'recursive': False},
                                                        'SWAP_PIVOT': {'level': 0, 'recursive': False},
                                                        'PUSH': {'level': 0, 'recursive': False},
                                                        'POP': {'level': 0, 'recursive': False},
                                                        'SAVE_PTR_1': {'level': 0, 'recursive': False},
                                                        'LOAD_PTR_1': {'level': 0, 'recursive': False},
                                                        'DECREASE_CTR': {'level': 0, 'recursive': False},
                                                        'PARTITION_UPDATE': {'level': 1, 'recursive': False},
                                                        'PARTITION': {'level': 2, 'recursive': False},
                                                        'SAVE_LOAD_PARTITION': {'level': 3, 'recursive': False},
                                                        'QUICKSORT_UPDATE': {'level': 4, 'recursive': False},
                                                        'QUICKSORT_UPDATE_REC': {'level': 5, 'recursive': True},
                                                        'QUICKSORT': {'level': 6, 'recursive': False}}.items()))

programs_library_reduced_recursive = OrderedDict(sorted({'STOP': {'level': -1, 'recursive': False},
                                                        'PTR_1_LEFT': {'level': 0, 'recursive': False},
                                                        'PTR_2_LEFT': {'level': 0, 'recursive': False},
                                                        'PTR_3_LEFT': {'level': 0, 'recursive': False},
                                                        'PTR_1_RIGHT': {'level': 0, 'recursive': False},
                                                        'PTR_2_RIGHT': {'level': 0, 'recursive': False},
                                                        'PTR_3_RIGHT': {'level': 0, 'recursive': False},
                                                        'SWAP': {'level': 0, 'recursive': False},
                                                        'SWAP_PIVOT': {'level': 0, 'recursive': False},
                                                        'PUSH': {'level': 0, 'recursive': False},
                                                        'POP': {'level': 0, 'recursive': False},
                                                        'SAVE_PTR_1': {'level': 0, 'recursive': False},
                                                        'LOAD_PTR_1': {'level': 0, 'recursive': False},
                                                        'DECREASE_CTR': {'level': 0, 'recursive': False},
                                                        'PARTITION': {'level': 1, 'recursive': False},
                                                        'QUICKSORT_UPDATE': {'level': 2, 'recursive': False},
                                                        'QUICKSORT': {'level': 3, 'recursive': True}}.items()))

programs_library_with_arguments = OrderedDict(sorted({'STOP': {'level': -1, 'recursive': False, 'args': [0]},
                                                        'PTR_LEFT': {'level': 0, 'recursive': False, 'args': [1,2,3]},
                                                        'PTR_RIGHT': {'level': 0, 'recursive': False, 'args': [1,2,3]},
                                                        'SWAP': {'level': 0, 'recursive': False, 'args': [2]},
                                                        'PUSH': {'level': 0, 'recursive': False, 'args': [0]},
                                                        'POP': {'level': 0, 'recursive': False, 'args': [0]},
                                                        'SAVE_PTR': {'level': 0, 'recursive': False, 'args': [1]},
                                                        'LOAD_PTR': {'level': 0, 'recursive': False, 'args': [1]},
                                                        'PARTITION_UPDATE': {'level': 1, 'recursive': False, 'args': [0]},
                                                        'PARTITION': {'level': 2, 'recursive': False, 'args': [0]},
                                                        'SAVE_LOAD_PARTITION': {'level': 3, 'recursive': False, 'args': [0]},
                                                        'QUICKSORT_UPDATE': {'level': 4, 'recursive': False, 'args': [0]},
                                                        'QUICKSORT': {'level': 5, 'recursive': False,'args': [0]}}.items()))


def generate_all_combinations(args):
    total_args = []
    for e in args:
        if e == 0:
            total_args.append([0, 0, 0])
        elif e == 1:
            total_args.append([1, 0, 0])
            total_args.append([0, 1, 0])
            total_args.append([0, 0, 1])
        elif e == 2:
            total_args.append([1, 0, 1])
            total_args.append([1, 1, 0])
            total_args.append([0, 1, 1])
        else:
            total_args.append([1, 1, 1])
    return total_args

def generate_total_possible_programs(program_dict):
    final_dict = {}

    for k in program_dict:
        args = program_dict[k]["args"]
        total_args = generate_all_combinations(args)
        for arg in total_args:
            # Add a new program which contains the current arguments
            final_dict[k+"_"+str(arg)] = {"level": program_dict[k]["level"],
                                      "recursive": program_dict[k]["recursive"],
                                      "args": program_dict[k]["args"],
                                      "determ_args": arg}
    return final_dict

def generate_total_possible_conditions(prog_to_func, precond, postcond, failed_envs, program_dict):

    for k in program_dict:
        new_k = k[:-10]

        if new_k in prog_to_func:
            prog_to_func[k] = prog_to_func[new_k]

        if new_k in precond:
            precond[k] = precond[new_k]

        if new_k in postcond:
            postcond[k] = postcond[new_k]

        if new_k in failed_envs:
            failed_envs[k] = []

    return prog_to_func, precond, postcond, failed_envs


def assert_partition_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp):
    assert init_pointers_pos3 < init_pointers_pos2 \
           and init_pointers_pos1 < init_pointers_pos2 \
           and init_pointers_pos1 <= init_pointers_pos3 and temp[0] != -1, "Partition update {}, {}, {}, {}".format(
        init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, temp)

def assert_partition(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp):
    assert init_pointers_pos3 == init_pointers_pos1 \
           and init_pointers_pos1 < init_pointers_pos2 \
           and temp[0] == init_pointers_pos1, "Partition {}, {}, {}, {}".format(
        init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, temp)

def assert_save_load_partition(init_pointers_pos1, init_pointers_pos2):
    assert init_pointers_pos1 < init_pointers_pos2 and init_pointers_pos1 == init_pointers_pos3, "Save Load Partition {}, {}".format(init_pointers_pos1, init_pointers_pos2)

def random_push(init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_temp_variables, init_prog_stack):

    if init_pointers_pos1 + 1 < init_pointers_pos2:
        init_prog_stack.append(init_pointers_pos1 + 1)
        init_prog_stack.append(init_pointers_pos2)
        init_prog_stack.append(init_pointers_pos1 + 1)

    if init_pointers_pos1 - 1 > 0 and init_pointers_pos3 < init_pointers_pos1 - 1:
        init_prog_stack.append(init_pointers_pos3)
        init_prog_stack.append(init_pointers_pos1 - 1)
        init_prog_stack.append(init_pointers_pos3)

    return init_prog_stack.copy()


def assert_quicksort_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp):
    assert len(stack) >= 3 and temp[0] == -1, "Quicksort Update: {}".format(stack)

def partition_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp, counter, sampled_environment={}, sample=True):

    """ (3)
    Representation as sub commands:
    * SWAP_PIVOT
    * PTR_1_RIGHT
    * STOP

    :param scratchpad_ints:
    :param init_pointers_pos1:
    :param init_pointers_pos2:
    :param init_pointers_pos3:
    :param stack:
    :param temp:
    :param stop:
    :param stop_partition_update:
    :return:
    """

    if sample:
        sampled_environment["PARTITION_UPDATE"].append((np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack.copy(), temp.copy(), counter))

    if scratchpad_ints[init_pointers_pos3] < scratchpad_ints[init_pointers_pos2]:
        scratchpad_ints[[init_pointers_pos3, init_pointers_pos1]] = scratchpad_ints[
            [init_pointers_pos1, init_pointers_pos3]]
        init_pointers_pos1 += 1

    return np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack.copy(), temp.copy(), counter


def partition(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp, counter, sampled_environment={}, sample=True):
    """
    (total of 2*(n-1)+2)
    from 0 to n-1:
        PARTITION_UPDATE  *OR*  SWAP_PIVOT
                                PTR_1_RIGHT
        PTR_3_RIGHT
    SWAP
    STOP

    :param scratchpad_ints:
    :param init_pointers_pos1:
    :param init_pointers_pos2:
    :param init_pointers_pos3:
    :param stack:
    :param temp:
    :param stop:
    :param stop_partition:
    :param stop_partition_update:
    :return:
    """

    if sample:
        sampled_environment["PARTITION"].append((np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack.copy(), temp.copy(), counter))

    while init_pointers_pos3 < init_pointers_pos2:
        scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp, counter = \
        partition_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp, counter, sampled_environment=sampled_environment, sample=sample)
        init_pointers_pos3 += 1

    scratchpad_ints[[init_pointers_pos1, init_pointers_pos2]] = scratchpad_ints[[init_pointers_pos2, init_pointers_pos1]]

    return np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack.copy(), temp.copy(), counter

def save_load_partition(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack, init_temp_variables, counter, sampled_environment={}, sample=True):

    """ (4 operations) or (3+ 3*(n-1)+1)
    SAVE_PTR1
    PARTITION
    LOAD_PTR1
    STOP

    :param scratchpad_ints:
    :param init_pointers_pos1:
    :param init_pointers_pos2:
    :param init_pointers_pos3:
    :param init_prog_stack:
    :param init_temp_variables:
    :param stop:
    :param stop_partition:
    :param stop_partition_update:
    :param stop_save_load_partition:
    :return:
    """

    if sample:
        sampled_environment["SAVE_LOAD_PARTITION"].append((np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack.copy(), init_temp_variables.copy(), counter))

    init_temp_variables = [init_pointers_pos1]

    # Run the partition method
    scratchpad_ints, init_pointers_pos1, init_pointers_pos2, \
    init_pointers_pos3, init_prog_stack, init_temp_variables, counter = \
        partition(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack,
                init_temp_variables, counter, sampled_environment, sample=sample)

    init_pointers_pos3 = init_temp_variables[0]
    init_temp_variables = [-1]

    return np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack.copy(), init_temp_variables.copy(), counter

def quicksort_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack, init_temp_variables, counter, sampled_environment={}, sample=True):

    """ (4 operations) or 7 (if we do not use the save load partition)
    POP
    SAVE_LOAD_PARTITION
    DECREASE_CTR (only if in recursive mode)
    PUSH
    STOP

    POP
    SAVE_PTR1
    PARTITION
    LOAD_PTR1
    DECREASE_CTR
    PUSH
    STOP

    :param scratchpad_ints:
    :param init_pointers_pos1:
    :param init_pointers_pos2:
    :param init_pointers_pos3:
    :param init_prog_stack:
    :param init_temp_variables:
    :param stop:
    :param stop_partition:
    :param stop_partition_update:
    :param stop_quicksort_update:
    :return:
    """

    if sample:
        sampled_environment["QUICKSORT_UPDATE"].append((np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack.copy(), init_temp_variables.copy(), counter))

    init_pointers_pos1 = init_prog_stack.pop()
    init_pointers_pos2 = init_prog_stack.pop()
    init_pointers_pos3 = init_prog_stack.pop()

    if init_pointers_pos1 < init_pointers_pos2:

        scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack, init_temp_variables, counter = \
            save_load_partition(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3,
                            init_prog_stack, init_temp_variables, counter, sampled_environment, sample=sample)

        init_prog_stack = random_push(init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_temp_variables.copy(), init_prog_stack.copy())

    return np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack.copy(), init_temp_variables.copy(), counter-1

def sample_quicksort_indexes(scratchpad_ints, length):

    """ (1+n+1) or (1+1+1) if the QUICKSORT_UPDATE function is recursive
    PUSH
    from 0 to n:
        QUICKSORT_UPDATE
    STOP

    :param scratchpad_ints:
    :param length:
    :param sort:
    :param stop_partition:
    :param stop_partition_update:
    :param stop_quicksort_update:
    :return:
    """

    sampled_environment = OrderedDict(sorted({"QUICKSORT": [],
                                              "PARTITION_UPDATE": [],
                                              "PARTITION": [],
                                              "SAVE_LOAD_PARTITION": [],
                                              "QUICKSORT_UPDATE_REC": [],
                                              "QUICKSORT_UPDATE": []}.items()))

    init_pointers_pos1 = 0
    init_pointers_pos2 = length - 1
    init_pointers_pos3 = 0

    init_temp_variables = [-1]

    init_prog_stack = []
    init_prog_stack.append(init_pointers_pos3)
    init_prog_stack.append(init_pointers_pos2)
    init_prog_stack.append(init_pointers_pos1)

    counter = length

    sampled_environment["QUICKSORT"].append((np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack.copy(), init_temp_variables.copy(), counter))
    sampled_environment["QUICKSORT_UPDATE_REC"].append((np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack.copy(), init_temp_variables.copy(), counter))

    while len(init_prog_stack) > 0:
        scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack, init_temp_variables, counter = \
           quicksort_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack, init_temp_variables, counter, sampled_environment)

    sampled_environment["QUICKSORT_UPDATE_REC"].append((np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack.copy(), init_temp_variables.copy(), counter))
    sampled_environment["QUICKSORT"].append((np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack.copy(), init_temp_variables.copy(), counter))

    return sampled_environment.copy()


# Testing
if __name__ == "__main__":
    for i in range(0,10000):

        arr = np.random.randint(0, 100, 7)
        print("")

        scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp, counter = \
        ([0, 6, 0, 6, 7, 7, 8], 4, 5, 0, [0,0,3], [-1], 5)
        print(quicksort_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp, counter, sample=False))
        break

        env = sample_quicksort_indexes(np.copy(arr), 7)

        for e in env["PARTITION_UPDATE"]:
            scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp = e
            assert_partition_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp)

        for e in env["PARTITION"]:
            scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp = e
            assert_partition(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp)

        for e in env["SAVE_LOAD_PARTITION"]:
            scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp = e
            assert_save_load_partition(init_pointers_pos1, init_pointers_pos2)

        for e in env["QUICKSORT_UPDATE"]:
            scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp = e
            assert_quicksort_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp)

        scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp = env["QUICKSORT"][1]

        if not np.all(scratchpad_ints[:len(scratchpad_ints) - 1] <= scratchpad_ints[1:len(scratchpad_ints)]):
            print("Not Sorted")

        if not np.array_equal(np.array(sorted(arr)), np.array(scratchpad_ints)):
            print("{}, {}".format(np.array(sorted(arr)), np.array(scratchpad_ints)))

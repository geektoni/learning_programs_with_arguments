import subprocess
import random

# default_config
# configs = ["complete", "without-partition-update", "reduced", "recursive"]
# train_errors = [True, False]
# expose_pointers = [True, False]
# output_tb_dir = "./final_results_tb"

# set random seed
random.seed(42)

configs = [("complete", -1, "", 1),
           ("without-partition-update", -1, "", 1),
           ("without-save-load-partition", -1, "", 1),
           ("reduced", -1, "", 1)]
train_errors = [0.3]
seeds = random.sample(range(0, 10000), 3)
use_complete_actions = [False, True]
dirichlet_noise = [0.9]
dirichlet_eps = [0.3, 0.4]
widening = [True, False]
max_exploration_nodes = [1, 0.8, 0.6, 0.4]
default_childs = [5, 20, 30]
use_gpu = False
output_tb_dir = "/home/giovanni.detoni/alphanpi_models/models_widening_quicksort_2/training_tb"
output_model_dir = "/home/giovanni.detoni/alphanpi_models/models_widening_quicksort_2/models"
start_nancheck = -1
end_nancheck = 10000

for c in configs:
    for t in train_errors:
        for ca in use_complete_actions:
            for d in dirichlet_noise:
                for eps in dirichlet_eps:
                    for m_nodes in max_exploration_nodes:
                        for def_child in default_childs:
                            for wide in widening:
                                for s in seeds:

                                    if ca:
                                        output_tb_dir_target = output_tb_dir+"_all_actions"
                                    else:
                                        output_tb_dir_target = output_tb_dir

                                    if c[2] == "":
                                        model_name = "none"
                                    else:
                                        model_name = c[2]

                                    command = "bash submit_jobs.sh {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(
                                        c[0], t, output_tb_dir_target, s, d, ca, eps, output_model_dir, start_nancheck, end_nancheck,
                                        c[1], model_name, c[3], m_nodes, def_child, use_gpu, wide
                                    )

                                    # execute the command
                                    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                                    output, error = process.communicate()
                                    print(output.decode('UTF-8'))
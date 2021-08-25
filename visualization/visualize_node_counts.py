import argparse
import pandas as pd
import seaborn as sns
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import re

from tqdm import tqdm

sns.set(context="paper", style="ticks")
sns.set_palette('colorblind')

class MathTextSciFormatter(ticker.Formatter):
    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt
    def __call__(self, x, pos=None):
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
        return "${}$".format(s)

def get_validation_data(path, target_op):

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    complete_val = pd.DataFrame(columns=["Iteration", "Model", "Sampling",
                                         "Program", "Childs", "Validation Value"])

    for file_name in onlyfiles:

        print("[*] Processing {}".format(file_name))

        values = file_name.split("-")[1:]

        date = values[0]
        time_ = values[1]
        seed = values[2]
        str_c = values[3].lower() == "true"
        pen_level_0 = values[4].lower() == "true"
        leve_0_pen = float(values[5])
        expose_stack = values[6].lower() == "true"
        samp_err_poss = float(values[7])
        without_partition = values[8].lower() == "true"
        reduced_operation = values[9].lower() == "true"
        keep_training = values[10].lower() == "true"
        recursive_quicksort = values[11].lower() == "true"
        do_not_expose_pointer_values = values[12].lower() == "true"
        complete_actions = values[13].lower() == "true"
        dir_noise = float(values[14])
        dir_eps = float(values[15])
        normalize_policy = values[16].lower() == "true"
        without_save_load_partition = values[17].lower() == "true"
        default_childs = str(values[18])
        widening = values[19].lower() == "true"

        if not widening:
            default_childs = "No Sampling"

        operation_name = values[len(values)-1].replace("validation_", "").replace(".csv", "")

        if operation_name != target_op:
            continue

        model_type = "Complete" if not reduced_operation else "Reduced"
        if without_partition or without_save_load_partition:
            model_type = "No Partition Update" if without_partition else "No Save Load Partition"

        with open(join(path,file_name), 'r') as f:
            counter = 0
            skip_first = False
            for line in f:

                if not skip_first:
                    skip_first = True
                    continue

                validation_value = float(line.split(",")[2])

                complete_val = complete_val.append({
                    "Iteration": counter,
                    "Model": model_type,
                    "Sampling": widening,
                    "Program": operation_name,
                    "Childs": default_childs,
                    "Validation Value": validation_value
                }, ignore_index=True)

                counter += 1

    return complete_val

def convert_model_name(args, mcts):

    if args == "No Arguments":
        if mcts == "No Sampling":
            return r"$M_{alphanpi}$"
        else:
            return r"$M_{alphanpi}^S$"
    else:
        if mcts == "No Sampling":
            return r"$M_{alphaargs}$"
        else:
            return r"$M_{alphaargs}^S$"

if __name__ == "__main__":

    # Get command line params
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help='Path to the result file', default="../node_counts_paper", type=str)
    parser.add_argument("--path-validation", help='Path to the validation files', default="../validations_paper", type=str)
    parser.add_argument("--operation", help="Program to analyze", default="PARTITION_UPDATE", type=str)
    parser.add_argument("--max-iter", help="Max iterations", type=int, default=1000)
    parser.add_argument("--split-files", help="Consider the case in which we have split files.", action="store_true", default=False)
    parser.add_argument("--save", help="Save the result as image", action="store_true", default=False)
    parser.add_argument("--with-title", help="Add an explicative title to the graph", action="store_true", default=False)

    args = parser.parse_args()
    path = args.path
    path_validation = args.path_validation
    operation_to_test = args.operation
    max_iteration = args.max_iter
    split_files = args.split_files
    save_graph = args.save
    with_title = args.with_title

    if split_files:
        vd = get_validation_data(path_validation, operation_to_test)

    node_data = {"Iteration": [], "Model Name":[],  "Sampling": [], "Program":[],
                 "# Node Expanded":[], "# Sampled Nodes":[], "# Node Selected / # Episodes": [],
                 "Validation Accuracy": [], "Model":[]}

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    for file_name in tqdm(onlyfiles):

        node_iterations = {}

        if file_name[0] == ".":
            continue

        values = file_name.split("-")
        date = values[0]
        time_ = values[1]
        seed = values[2]
        str_c = values[3].lower() == "true"
        pen_level_0 = values[4].lower() == "true"
        leve_0_pen = float(values[5])
        expose_stack = values[6].lower() == "true"
        samp_err_poss = float(values[7])
        without_partition = values[8].lower() == "true"
        reduced_operation = values[9].lower() == "true"
        keep_training = values[10].lower() == "true"
        recursive_quicksort = values[11].lower() == "true"
        do_not_expose_pointer_values = values[12].lower() == "true"
        complete_actions = values[13].lower() == "true"
        dir_noise = float(values[14])
        dir_eps = float(values[15])
        normalize_policy = values[16].lower() == "true"
        without_save_load_partition = values[17].lower() == "true"
        default_childs = str(values[18])
        widening = values[19].lower() == "true"

        if seed != "2022":
            continue

        model_type = "Complete" if not reduced_operation else "Reduced"
        if without_partition or without_save_load_partition:
            model_type = "No Partition Update" if without_partition else "No SL Partition"

        if complete_actions:
            #model_type += " (No Args)"
            model_type = "No Arguments"
        else:
            model_type = "With Arguments"

        with open(join(path,file_name), 'r') as f:
            for line in f:

                # Prevent wrong splitting
                if complete_actions:
                    line = line.replace("_[0, 0, 0]", "")

                # 0: iteration
                # 1: operation: nodes expanded - mean node selected
                content_str = line.split(",")

                index_node_selected = 2 if split_files else 3

                regexp_numbers = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
                content = [float(x) for x in re.findall(regexp_numbers, line)]

                iteration = int(content[0])
                operation = content_str[1].split(":")[0]
                node_expanded = int(content[1])
                node_selected = int(abs(content[index_node_selected]))

                signature = "{}-{}-{}-{}".format(model_type, widening, operation, default_childs)

                #if operation != operation_to_test:
                #    continue

                if signature in node_iterations:
                    iteration, expanded_count = node_iterations[signature]
                    node_iterations[signature] = (iteration+1, expanded_count+node_expanded)
                else:
                    node_iterations[signature] = (0, node_expanded)
                    iteration = 0
                    expanded_count = node_expanded

                if iteration > 5000:
                    break

                if not widening:
                    default_childs = "No Sampling"

                node_data["Iteration"].append(iteration)
                node_data["Model Name"].append(model_type)
                #node_data["Model"].append(model_type + " (MCTS)" if default_childs == "No Sampling" else model_type + " (A-MCTS)")
                node_data["Model"].append(convert_model_name(model_type, default_childs))
                node_data["Sampling"].append(widening)
                node_data["# Sampled Nodes"].append(default_childs)
                node_data["Program"].append(operation)
                node_data["# Node Expanded"].append(expanded_count)
                node_data["# Node Selected / # Episodes"].append(node_selected)

                if split_files:
                    val_result = vd[(vd.Iteration == iteration)
                       & (vd.Model == str(model_type))
                       & (vd.Program == str(operation))
                       & (vd.Childs == str(default_childs))
                       ]
                    if len(val_result) != 0:
                        node_data["Validation Accuracy"].append(val_result["Validation Value"].item())
                    else:
                        node_data["Validation Accuracy"].append(0.0)
                else:
                    val_accuracy = abs(content[2])
                    node_data["Validation Accuracy"].append(val_accuracy)


    # Convert the results into a dataframe
    df = pd.DataFrame(node_data)

    df = df[df.Iteration <= max_iteration]

    print("[*] Dataframe dimensions: ", len(df))
    print(df.head())

    g = sns.FacetGrid(df, col="Program", hue="Model")
    g.map(sns.lineplot, "Iteration", "# Node Expanded").set(yscale = 'log')
    #g.map(sns.lineplot, "Iteration", "Validation Accuracy")
    g.add_legend()

    # Correct the layout
    #g.fig.tight_layout()

    if not save_graph:
        plt.show()
    else:
        plt.savefig("{}-results.png".format(operation_to_test), dpi=300, bbox_inches='tight')

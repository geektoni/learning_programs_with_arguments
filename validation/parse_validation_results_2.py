import pandas as pd
import glob
import os
import itertools

import argparse
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

sns.set(context="paper", style="whitegrid")

def autolabel(rects, ax, ann_size):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=ann_size, fontweight="bold")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--op", type=str, default="PARTITION_UPDATE")
    parser.add_argument("--dir", type=str, default="../results")
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--annotate", action="store_true", default=False)
    parser.add_argument("--title", action="store_true", default=False)
    parser.add_argument("--legend", action="store_true", default=False)
    parser.add_argument("--annotation-size", type=int, default=10)
    parser.add_argument("--line", action="store_true", default=False)
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--net", action="store_true", default=False)
    parser.add_argument("--latex", action="store_true", default=False)
    parser.add_argument("--std", action="store_true", default=False)
    parser.add_argument("--failures", action="store_true", default=False)
    parser.add_argument("--exclude", nargs="?", help="Exclude model (complete, reduced, w_save, w_part)")
    parser.add_argument("--omit-model-name", action="store_true", default=False)
    args = parser.parse_args()

    result_files = glob.glob(args.dir + "/validation_*")
    operation_name = args.op
    omit_model_name = args.omit_model_name

    # exclude some models
    exclude_list = args.exclude if not args.exclude is None else []

    # Possible value combinations
    # reduced, no_part_upd, no_save_load, expose_stack, expose_pointers, complete_actions, sampling, retrain
    combinations = [[False, False, True, False, False, False, False, 0.35, 0.35, 0.2],
                    [False, False, True, False, False, False, True, 0.35, 0.35, 0.2],
                    [False, False, False, False, False, True, False, 0.35, 0.35, 0.2],
                    [False, False, False, False, False, True, True, 0.35, 0.35, 0.2]
                    ]

    order_operations = [
        "PARTITION_UPDATE",
        "PARTITION",
        "QUICKSORT_UPDATE",
        "QUICKSORT"
    ]

    total_results = []

    # Choose which measure whats to read (mcts or net)
    method = "mcts" if not args.net else "net"
    method_std = "mcts_std" if not args.net else "net_std"

    if args.failures:
        method = "failed_subprog"

    # For each file, parse its content and generate a table
    for f in result_files:

        file_name = os.path.basename(f)

        file_name_values = file_name.split("-")

        operation = file_name_values[2]
        prob_samb_err = file_name_values[3]
        reduced_operation = file_name_values[4] == "True"
        without_partition = file_name_values[5] == "True"
        expose_stack = file_name_values[6] == "True"
        recursive_quicksort = file_name_values[7] == "True"
        do_not_expose_pointers = file_name_values[8] == "True"

        if len(file_name_values) > 10:
            complete_actions = file_name_values[9] == "True"
            dir_noise = file_name_values[10]
            dir_eps = file_name_values[11]
            normalize_polices = file_name_values[12] == "True"
            without_save_load_partition = file_name_values[13] == "True"
            num_childs = int(file_name_values[14])
            widening = file_name_values[15] == "True"
            seed = file_name_values[16].split(".")[0].replace("\n", "")
            use_args = True
        else:
            complete_actions = True
            dir_noise = 0.35
            dir_eps = 0.3
            normalize_polices = False
            without_save_load_partition = False
            num_childs = 5
            widening = False
            seed = file_name_values[9].split(".")[0].replace("\n", "")
            use_args = False

        # Skip the models we do not want
        if reduced_operation and "reduced" in exclude_list:
            continue
        if without_partition and ("w_part" in exclude_list):
            continue
        if without_save_load_partition and "w_save" in exclude_list:
            continue
        if "complete" in exclude_list:
            continue

        model_type = "Complete" if not reduced_operation else "Reduced"
        if without_partition or without_save_load_partition:
            model_type = "No Partition Update" if without_partition else "No Save Load Partition"

        model_type = "With Arguments"

        if complete_actions:
            operation = operation[:-10]
            model_type = "No Arguments"

        if omit_model_name:
            if not complete_actions:
                model_type = "With Arguments"
            else:
                model_type = "Without Arguments"

        # File results
        results = [operation, float(prob_samb_err), model_type,
                   expose_stack,
                   recursive_quicksort, do_not_expose_pointers, complete_actions, float(dir_noise), float(dir_eps),
                   normalize_polices, num_childs, widening, float(seed), use_args]

        with open(f, "r") as open_file:

            skip = 0
            for line in open_file:
                if skip == 0:
                    skip += 1
                    continue

                # Split the various information
                # 0: length
                # 1: mcts mean
                # 2: mcts normalized
                # 3: net mean
                values = line.split(",")

                length = int(values[0].split(":")[1])
                mcts_norm = float(values[2].split(":")[1])
                net_mean = float(values[3].split(":")[1])
                mcts_std = float(values[4].split(":")[1])
                net_std = float(values[5].split(":")[1])

                if len(values) > 6:
                    failed_programs = float(values[6].split(":")[1])
                    failed_subprograms = float(values[7].split(":")[1].replace("\n", ""))
                else:
                    failed_subprograms = 0
                    failed_programs=0

                if failed_subprograms != 0:
                    percentage_of_failed_subprg = failed_subprograms
                else:
                    percentage_of_failed_subprg = 0.0

                #if length > 65:
                #    continue

                # Appent the final result
                total_results.append(
                    results + [length, mcts_norm, net_mean, mcts_std, net_std, percentage_of_failed_subprg,
                               failed_programs - failed_subprograms])

    # Generate the pandas dataframe
    df = pd.DataFrame(total_results,
                      columns=["operation", "samp_err", "model_type", "expose_stack",
                               "recursive",
                               "expose_pointers", "complete_actions", "dir_noise", "dir_eps", "normalize_policies",
                               "num_childs", "approximate MCTS", "seed", "args",
                               "len", "mcts", "net", "mcts_std", "net_std", "failed_subprog", "failed_progs"])
    df.sort_values(by=["operation", "len", "samp_err"], inplace=True)

    df = df.rename(columns={method: 'accuracy'})

    g = sns.FacetGrid(df, col="operation", row="model_type", hue="approximate MCTS",
                      margin_titles=True, legend_out=True,
                      col_order=order_operations)
    g.map(sns.lineplot, "len", "accuracy")
    g.add_legend()

    #sns.lineplot(y=method, x="len", data=df[df.operation == operation_name], hue="model_type",
    #             style="widening", markers=True, dashes=False, ci="sd")#, estimator=None, lw=1, units="seed")

    # cosmetics

    #if not args.failures:
    #    plt.ylim(0, 1.1)

    if not args.failures:
        title = "Validation Accuracy {}".format(operation_name)
    else:
        title = "Failed Subprograms {}".format(operation_name)

    #plt.xlabel("List Length")
    #plt.ylabel("Accuracy")

    #plt.title(title)
    if args.title:
        g.fig.subplots_adjust(top=0.8)
        g.fig.suptitle(title)

    #plt.tight_layout()

    for op in ["PARTITION_UPDATE", "PARTITION", "QUICKSORT_UPDATE", "QUICKSORT"]:
        print(op)
        for v in [5, 10, 20, 40, 60]:
            only_op = df[(df["operation"] == op) & (df["len"] == v) & (df["seed"] == 2022)]
            only_op.sort_values(by=["model_type", "approximate MCTS"], inplace=True)
            print(only_op[["model_type","approximate MCTS", "accuracy", "mcts_std"]])
            print()


    # Show the plot
    if not args.save:
        plt.show()
    else:
        if args.net:
            name = "net"
        else:
            name = "mcts"

        if args.failures:
            name += "_failures"

        plt.savefig("{}_plot_{}.png".format(args.op, name), dpi=300, bbox_inches="tight")

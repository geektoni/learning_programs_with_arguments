from tensorflow.python.summary.summary_iterator import summary_iterator
import argparse
import glob
import pandas as pd
import pprint

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook")
sns.set_palette(sns.color_palette("RdBu_r"))

import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--op", type=str, default="PARTITION_UPDATE")
    parser.add_argument("--dir", type=str, default="../results")
    parser.add_argument("--file", type=str, default="complete_tensorboard_log.csv")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--show", default=False, action="store_true")
    args = parser.parse_args()

    skip_file_generation = False
    if os.path.exists(args.file):
        skip_file_generation = True

    result_files = glob.glob(args.dir + "/*/events.out*")

    df = pd.DataFrame(columns=['operation', "model_type", 'step', 'epsilon', 'noise', 'seed', 'accuracy'])

    possible_eps = []
    possible_noise = []
    possible_seed = []
    possible_models = []

    for f in result_files:

        directory_name = os.path.basename(os.path.dirname(f))

        #print("[*] Parsing file {}".format(directory_name))

        file_name_values = directory_name.split("-")

        seed = file_name_values[2]
        dir_eps = file_name_values[15]
        dir_noise = file_name_values[14]
        reduced_operation = file_name_values[9] == "True"

        model_type = "complete" if not reduced_operation else "reduced"

        if not dir_eps in possible_eps:
            possible_eps.append(dir_eps)

        if not dir_noise in possible_noise:
            possible_noise.append(dir_noise)

        if not seed in possible_seed:
            possible_seed.append(seed)

        if not model_type in possible_models:
            possible_models.append(model_type)

        step=0

        for summary in summary_iterator(f):

            if skip_file_generation:
                break

            tagval = []
            for e in summary.summary.value:
                    tag = e.tag.split("/")
                    if tag[0] == "validation" and tag[1] == "PARTITION":
                        operation = tag[1]
                        value = float(e.simple_value)
                        record = {'operation': operation,
                                  'model_type': model_type,
                                  'step': int(step),
                                  'epsilon': float(dir_eps),
                                  'noise': float(dir_noise),
                                  'seed': int(seed),
                                  'accuracy': float(value)}
                        df = df.append(
                            record ,
                            ignore_index=True
                        )
                        step += 1

    if skip_file_generation:
        df = pd.read_csv("complete_tensorboard_log.csv")
    else:
        df.to_csv("complete_tensorboard_log.csv")

    # Compute the best hyper
    results = []

    for m in possible_models:
        for s in possible_seed:
            for n in possible_noise:
                for e in possible_eps:
                    current = df[(df.noise == float(n)) & (df.epsilon == float(e)) & (df.seed == int(s)) & (df.model_type == m)]
                    mean = current["accuracy"][-args.k:].mean()
                    results.append([m, s, n, e, mean])

    sorted_result = sorted(results, key= lambda r: r[4], reverse=True)
    sorted_result = pd.DataFrame(sorted_result, columns=["model", "seed", "noise", "eps", "mean_acc"])

    pprint.pprint(sorted_result)

    if args.show:
        best_results = pd.DataFrame(columns=['operation', "model_type", 'step', 'epsilon', 'noise', 'seed', 'accuracy'])

        for m in possible_models:
            best_model_results = sorted_result[sorted_result.model == m]
            counter=0
            for index, row in best_model_results.iterrows():
                if counter >= args.top:
                    break
                model, seed, noise, epsilon = row["model"], row["seed"], row["noise"], row["eps"]
                current = df[(df.noise == float(noise)) & (df.epsilon == float(epsilon)) & (df.seed == int(seed)) & (df.model_type == model)][0:300]
                best_results = pd.concat([best_results, current], sort=False)
                counter += 1

        unique_hue_values = len(best_results.epsilon.unique())

        sns.relplot(y="accuracy", x="step", data=best_results, col="model_type",
                     hue="epsilon", style="noise", markers=False, dashes=True, kind="line",
                    palette=sns.color_palette("muted", n_colors=unique_hue_values))  # , estimator=None, lw=1, units="seed")
        plt.show()
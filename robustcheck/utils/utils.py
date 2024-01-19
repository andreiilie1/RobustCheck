from matplotlib import pyplot as plt
import numpy as np
import json
import mlflow
import os

PRINT_SEPARATOR = "_" * 19


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def save_evoba_artifacts(evoba_stats, run_output_folder):
    with open(run_output_folder + "/evoba_l0_stats.json", "w") as outfile:
        json.dump(dict(evoba_stats), outfile, cls=NpEncoder)

    np.save(run_output_folder + "/evoba_l0_stats.npy", evoba_stats)

    fig = plt.figure(figsize=(20, 14))
    plt.hist(evoba_stats["l0_dists_succ"])
    plt.title("EvoBA L0 distances histogram", fontsize=26)
    plt.xlabel("L0 distance", fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel("Count images", fontsize=24)
    plt.savefig(run_output_folder + "/evoba_l0_hist.png")

    fig = plt.figure(figsize=(20, 14))
    plt.hist(evoba_stats["queries_succ"])
    plt.title("EvoBA queries histogram", fontsize=26)
    plt.xlabel("Queries", fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel("Count images", fontsize=24)
    plt.savefig(run_output_folder + "/evoba_l0_queries_hist.png")


def save_robustness_stats_artifacts(robustness_check, run_output_folder):
    robustness_stats = robustness_check.get_stats()

    with open(run_output_folder + "/robustness_stats.json", "w") as outfile:
        json.dump(dict(robustness_stats), outfile, cls=NpEncoder)

    fig = plt.figure(figsize=(20, 14))
    plt.hist(robustness_stats["l2_dists_succ"])
    plt.title("L2 distances histogram", fontsize=26)
    plt.xlabel("L2 distance", fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel("Count images", fontsize=24)
    plt.savefig(run_output_folder + "/l2_hist.png")

    fig = plt.figure(figsize=(20, 14))
    plt.hist(robustness_stats["queries_succ"])
    plt.title("SimBA queries histogram", fontsize=26)
    plt.xlabel("Queries", fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel("Count images", fontsize=24)
    plt.savefig(run_output_folder + "/queries_hist.png")


def save_histogram(values, fname, title, clf=True, figsize=(20, 14)):
    fig = plt.figure(figsize=figsize)
    plt.title(title)
    plt.hist(values)
    plt.savefig(fname, bbox_inches="tight")

    if clf:
        plt.clf()

    return fig

def generate_mlflow_logs(robustness_check, run_name, experiment_name="default", tracking_uri="mlruns"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=run_name)

    robustness_metrics = robustness_check.get_stats()

    for metric in robustness_metrics:
        metric_value = robustness_metrics[metric]
        # Only log non-array and non-list stats, as robustness_metrics can also contain lists or arrays that are not
        #  supported by mlflow.log_metric
        if type(metric_value) not in [list, np.array]:
            mlflow.log_metric(metric, robustness_metrics[metric])

    l0_dists_succ = robustness_metrics["l0_dists_succ"]
    l2_dists_succ = robustness_metrics["l0_dists_succ"]
    queries_succ = robustness_metrics["queries_succ"]

    l0_dists_hist_fname = "l0_dists_histogram.png"
    _ = save_histogram(
        values=l0_dists_succ,
        fname=l0_dists_hist_fname,
        title="L0 norm distribution of successful perturbations",
        clf=True,
    )
    mlflow.log_artifact(l0_dists_hist_fname)
    os.remove(l0_dists_hist_fname)

    l2_dists_hist_fname = "l2_dists_histogram.png"
    _ = save_histogram(
        values=l2_dists_succ,
        fname=l2_dists_hist_fname,
        title="L2 norm distribution of successful perturbations",
        clf=True,
    )
    mlflow.log_artifact(l2_dists_hist_fname)
    os.remove(l2_dists_hist_fname)

    queries_hist_fname = "queries_histogram.png"
    _ = save_histogram(
        values=queries_succ,
        fname=queries_hist_fname,
        title="Query counts of successful perturbations",
        clf=True,
    )
    mlflow.log_artifact(queries_hist_fname)
    os.remove(queries_hist_fname)

    adversarial_strategy_indices = robustness_check.get_adversarial_strategy_indices()

    for i in adversarial_strategy_indices:
        if robustness_check.get_adversarial_strategy_perturbed_flag(i):
            fname = f"{i}_perturbed_succ.png"
        else:
            fname = f"{i}_perturbed_fail.png"

        perturbed_img = robustness_check.get_adversarial_strategy_perturbed_image(i)

        plt.imsave(fname, perturbed_img / 255)
        mlflow.log_artifact(fname)
        os.remove(fname)

        fname = f"{i}_original.png"
        plt.imsave(fname, robustness_check.x_test[i] / 255)
        mlflow.log_artifact(fname)
        os.remove(fname)

    mlflow.end_run()

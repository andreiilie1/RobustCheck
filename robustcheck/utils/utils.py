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


def save_histogram(values, fname, title, x_label="", y_label="", clf=True, fig_size=(20, 14), font_size=24):
    fig = plt.figure(figsize=fig_size)
    plt.title(title)

    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)

    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.hist(values)
    plt.savefig(fname, bbox_inches="tight")

    if clf:
        plt.clf()

    return fig


def save_robustness_stats_artifacts(robustness_check, run_output_folder):
    """
    Saves robustness check artifacts containings metrics and histograms of queries and perturbartion distances on the
    local file system.

    Attributes:
        robustness_check: RobustnessCheck containing the model and dataset to be benchmarked. This requires its
            run_robustness_check() method to have been executed such that we have metrics to extract from it.
        run_output_folder: A string representing where to save the arising artifacts.
    """
    robustness_stats = robustness_check.get_stats()

    with open(run_output_folder + "/robustness_stats.json", "w") as outfile:
        json.dump(dict(robustness_stats), outfile, cls=NpEncoder)

    l0_dists_succ = robustness_stats["l0_dists_succ"]
    l2_dists_succ = robustness_stats["l0_dists_succ"]
    queries_succ = robustness_stats["queries_succ"]

    l0_dists_hist_fname = "l0_dists_histogram.png"
    _ = save_histogram(
        values=l0_dists_succ,
        fname=os.path.join(run_output_folder, l0_dists_hist_fname),
        title="L0 distance distribution of successful perturbations",
        x_label="L0 distance",
        y_label="Image count",
        clf=False,
    )

    l2_dists_hist_fname = "l2_dists_histogram.png"
    _ = save_histogram(
        values=l2_dists_succ,
        fname=os.path.join(run_output_folder, l2_dists_hist_fname),
        title="L2 distance distribution of successful perturbations",
        x_label="L2 distance",
        y_label="Image count",
        clf=False,
    )

    queries_hist_fname = "queries_histogram.png"
    _ = save_histogram(
        values=queries_succ,
        fname=os.path.join(run_output_folder, queries_hist_fname),
        title="Query count distribution of successful perturbations",
        x_label="Query count",
        y_label="Image count",
        clf=False,
    )


def generate_mlflow_logs(robustness_check, run_name, experiment_name="default", tracking_uri="mlruns"):
    """
    Generates robustness check logs on mlflow.

    Arguments:
        robustness_check: RobustnessCheck containing the model and dataset to be benchmarked. This requires its
            run_robustness_check() method to have been executed such that we have metrics to extract from it.
        run_name: A string representing the run name under which the mlflow artifacts and metrics will be logged.
        experiment_name: A string representing the experiment name under which the mlflow artifacts and metrics will be
            logged.
        tracking_uri: A string representing the path where the mlflow artifacts and metrics will be stored.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=run_name)

    robustness_stats = robustness_check.get_stats()

    for metric in robustness_stats:
        metric_value = robustness_stats[metric]
        # Only log non-array and non-list stats, as robustness_metrics can also contain lists or arrays that are not
        #  supported by mlflow.log_metric
        if type(metric_value) not in [list, np.array]:
            mlflow.log_metric(metric, robustness_stats[metric])

    l0_dists_succ = robustness_stats["l0_dists_succ"]
    l2_dists_succ = robustness_stats["l0_dists_succ"]
    queries_succ = robustness_stats["queries_succ"]

    l0_dists_hist_fname = "l0_dists_histogram.png"
    _ = save_histogram(
        values=l0_dists_succ,
        fname=l0_dists_hist_fname,
        title="L0 distance distribution of successful perturbations",
        x_label="L0 distance",
        y_label="Image count",
        clf=True,
    )
    mlflow.log_artifact(l0_dists_hist_fname)
    os.remove(l0_dists_hist_fname)

    l2_dists_hist_fname = "l2_dists_histogram.png"
    _ = save_histogram(
        values=l2_dists_succ,
        fname=l2_dists_hist_fname,
        title="L2 distance distribution of successful perturbations",
        x_label="L2 distance",
        y_label="Image count",
        clf=True,
    )
    mlflow.log_artifact(l2_dists_hist_fname)
    os.remove(l2_dists_hist_fname)

    queries_hist_fname = "queries_histogram.png"
    _ = save_histogram(
        values=queries_succ,
        fname=queries_hist_fname,
        title="Query counts of successful perturbations",
        x_label="Queries",
        y_label="Image count",
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

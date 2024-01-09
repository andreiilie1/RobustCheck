from matplotlib import pyplot as plt
import numpy as np
import json
import mlflow
import os
import math
from robustcheck.types import AttackType

PRINT_SEPARATOR = "_" * 19


def cap(value, inf, sup):
    if value <= inf:
        return inf
    if value >= sup:
        return sup
    return value


def get_evoba_stats(adv_evo_strategy: dict):
    count_succ = 0
    queries_succ = []
    l0_dists_succ = []
    l2_dists_succ = []
    indices_succ = []

    count_fail = 0
    indices_fail = []

    adv_evo_strategy_keys = list(adv_evo_strategy.keys())

    for i in adv_evo_strategy_keys:
        img = adv_evo_strategy[i].img

        if adv_evo_strategy[i].is_perturbed():
            count_succ += 1
            queries_succ.append(adv_evo_strategy[i].queries)
            l0_dists_succ.append(
                np.sum(adv_evo_strategy[i].get_best_candidate() != img)
            )

            curr_l2 = np.sqrt(
                np.sum((adv_evo_strategy[i].get_best_candidate() - img) ** 2)
            )
            l2_dists_succ.append(curr_l2)

            indices_succ.append(i)
        else:
            count_fail += 1
            indices_fail.append(i)

    img_shape = np.shape(adv_evo_strategy[adv_evo_strategy_keys[0]].img)
    count_px = img_shape[0] * img_shape[1] * img_shape[2]

    # Will report l2 distances on [0,1] pixel scale, as this is usual in the literature
    # e.g. ImageNet is on [0,255]. Note l0 doesn't need to be normalised, as it's a count
    img_scale = adv_evo_strategy[adv_evo_strategy_keys[0]].pixel_space_max

    return {
        "count_succ": int(count_succ),
        "queries_succ": queries_succ,
        "l0_dists_succ": l0_dists_succ,
        "l2_dists_succ": l2_dists_succ,
        "indices_succ": indices_succ,
        "count_fail": int(count_fail),
        "indices_fail": indices_fail,
        "queries_succ_mean": np.mean(queries_succ),
        "l0_dists_succ_mean": np.mean(l0_dists_succ),
        "l2_dists_succ_mean": np.mean(l2_dists_succ) / img_scale,
        "l2_dists_succ_mean_pp": np.mean(l2_dists_succ) / (count_px * img_scale),
    }


def print_evoba_stats(evoba_stats):
    SEP = "_" * 20
    count_succ = evoba_stats["count_succ"]
    count_fail = evoba_stats["count_fail"]
    count_total = count_succ + count_fail

    queries_succ_mean = evoba_stats["queries_succ_mean"]
    l0_dists_succ_mean = evoba_stats["l0_dists_succ_mean"]

    queries_succ = evoba_stats["queries_succ"]
    l0_dists_succ = evoba_stats["l0_dists_succ"]

    l2_dists_succ_mean_pp = evoba_stats["l2_dists_succ_mean_pp"]

    print()
    print("EvoBA STATS (L0 attack)")
    print(SEP)

    print(f"Perturbed successfully {count_succ}/{count_total} images")
    print(f"Average query count: {queries_succ_mean}")
    print(f"Average l0 distance: {l0_dists_succ_mean}")
    print(f"Average l2 distance per pixel: {l2_dists_succ_mean_pp}")

    print()
    print(f"Median query count: {np.median(queries_succ)}")
    print(f"Median l0 dist: {np.median(l0_dists_succ)}")

    print()
    print(f"Max query count: {max(queries_succ)}")
    print(f"Max l0 dist: {max(l0_dists_succ)}")
    print(SEP)
    print()


def get_epsgreedy_stats(ega: dict):
    l0_dists_succ = []
    queries_succ = []
    indices_succ = []
    indices_fail = []
    for i in ega.keys():
        if ega[i].is_perturbed():
            dist = (ega[i].img != ega[i].get_best_candidate()).sum()
            l0_dists_succ.append(dist)
            queries_succ.append(ega[i].queries)
            indices_succ.append(i)
        else:
            indices_fail.append(i)

    return {
        "count_succ": len(queries_succ),
        "queries_succ": queries_succ,
        "l0_dists_succ": l0_dists_succ,
        "indices_succ": indices_succ,
        "count_fail": len(indices_fail),
        "indices_fail": indices_fail,
        "queries_succ_mean": np.mean(queries_succ),
        "l0_dists_succ_mean": np.mean(l0_dists_succ),
    }


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


def save_original_perturbed_image_pairs(adv_evo_strategy):
    # TODO: add implementation that saves pairs of original and
    #  perturbed images, together with individual stats
    #  Decide whether some sampling should happen here, or in
    #  the caller - maybe in the caller is a better idea.
    return "NOT IMPLEMENTED"


def get_simba_stats(simba_wrapper):
    count_succ = sum(simba_wrapper.perturbed)
    cont_fail = len(simba_wrapper.perturbed) - count_succ

    queries_succ = list(np.array(simba_wrapper.queries)[simba_wrapper.perturbed])
    l2_dists_succ = list(np.array(simba_wrapper.l2_distances)[simba_wrapper.perturbed])

    return {
        "count_succ": int(count_succ),
        "count_fail": int(cont_fail),
        "queries_succ": queries_succ,
        "l2_dists_succ": l2_dists_succ,
        "queries_succ_mean": np.mean(queries_succ),
        "l2_dists_succ_mean": np.mean(l2_dists_succ),
    }


def print_simba_stats(simba_stats):
    SEP = "_" * 20
    count_succ = simba_stats["count_succ"]
    count_fail = simba_stats["count_fail"]
    count_total = count_succ + count_fail

    queries_succ_mean = simba_stats["queries_succ_mean"]
    l2_dists_succ_mean = simba_stats["l2_dists_succ_mean"]

    queries_succ = simba_stats["queries_succ"]
    l2_dists_succ = simba_stats["l2_dists_succ"]

    print()
    print("SimBA STATS (L2 attack)")
    print(SEP)

    print(f"Perturbed successfully {count_succ}/{count_total} images")
    print(f"Average query count: {queries_succ_mean}")
    print(f"Average l2 distance: {l2_dists_succ_mean}")

    print()
    print(f"Median query count: {np.median(queries_succ)}")
    print(f"Median l2 dist: {np.median(l2_dists_succ)}")

    print()
    print(f"Max query count: {max(queries_succ)}")
    print(f"Max l2 dist: {max(l2_dists_succ)}")
    print(SEP)
    print()


def save_simba_artifacts(simba_stats, run_output_folder):
    with open(run_output_folder + "/simba_l2_stats.json", "w") as outfile:
        json.dump(dict(simba_stats), outfile, cls=NpEncoder)

    np.save(run_output_folder + "/simba_l2_stats.npy", simba_stats)

    fig = plt.figure(figsize=(20, 14))
    plt.hist(simba_stats["l2_dists_succ"])
    plt.title("SimBA L2 distances histogram", fontsize=26)
    plt.xlabel("L2 distance", fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel("Count images", fontsize=24)
    plt.savefig(run_output_folder + "/simba_l2_hist.png")

    fig = plt.figure(figsize=(20, 14))
    plt.hist(simba_stats["queries_succ"])
    plt.title("SimBA queries histogram", fontsize=26)
    plt.xlabel("Queries", fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel("Count images", fontsize=24)
    plt.savefig(run_output_folder + "/simba_l2_queries_hist.png")


def clear_mlflow():
    try:
        mlflow.end_run()
        print("Ended previous run")
    except Exception as e:
        print("Could not end a previous mlflow run - either an error occured or there is no previous mlflow run.")
        print("Error encountered: ", e)
        pass


def start_mlflow(run_name, experiment_name):
    try:
        print(f"Logging run {run_name} under experiment {experiment_name}")
        mlflow.set_experiment(experiment_name)
        mlflow.set_tracking_uri("/home/ailie/Repos/BBAttacks/mlruns/")
        mlflow.start_run(run_name=run_name)
    except Exception as e:
        print(e)
        print("Cannot start a new mlflow run, aborting")
        return "FAIL"


# TODO: move all mlflow logic to the RobustnessCheck class/helper class associated with it. Make these me
def generate_mlflow_logs(
        strategy_objects: dict,
        attack_type: AttackType,
        unperturbed_images: list,
        run_name: str,
        experiment_name="/default",
        additional_params={},
):
    clear_mlflow()

    start_mlflow(run_name, experiment_name)

    if attack_type == AttackType.EPSGREEDY:
        metrics = get_epsgreedy_stats(strategy_objects)
    elif attack_type == AttackType.EVOBA:
        metrics = get_evoba_stats(strategy_objects)
    else:
        raise NotImplementedError(f"attack_type {attack_type} not supported")

    l0_dists = metrics["l0_dists_succ"]
    queries_succ = metrics["queries_succ"]
    samples_succ = metrics["indices_succ"]
    samples_fail = metrics["indices_fail"]

    plt.title("L0's of successful perturbations")
    plt.hist(l0_dists)
    fname = "l0_dists.png"
    plt.savefig(fname, bbox_inches="tight")
    mlflow.log_artifact(fname)
    plt.clf()
    os.remove(fname)

    plt.title("Query counts of successful perturbations")
    plt.hist(queries_succ)
    fname = "queries_succ.png"
    plt.savefig(fname, bbox_inches="tight")
    mlflow.log_artifact(fname)
    plt.clf()
    os.remove(fname)

    for param in additional_params:
        mlflow.log_param(param, additional_params[param])

    mlflow.log_metric("l0_dists_suc_mean", np.mean(l0_dists))
    mlflow.log_metric("queries_suc_mean", np.mean(queries_succ))
    mlflow.log_metric("succes_rate", len(l0_dists) / len(strategy_objects))

    if attack_type == AttackType.EVOBA:
        mlflow.log_metric("l2_dists_succ_mean_pp", metrics["l2_dists_succ_mean_pp"])

    mlflow.log_param("perturbed", len(l0_dists))
    mlflow.log_param("images", len(strategy_objects))

    mlflow.log_param("l0_dists_suc", l0_dists)
    mlflow.log_param("queries_suc", queries_succ)
    mlflow.log_param("samples_succ", samples_succ)
    mlflow.log_param("samples_fail", samples_fail)

    sample_size = len(unperturbed_images)
    for i in range(sample_size):
        if strategy_objects[i].is_perturbed():
            fname = f"{i}_perturbed_succ.png"
        else:
            fname = f"{i}_perturbed_fail.png"

        if attack_type == AttackType.EPSGREEDY:
            img = strategy_objects[i].img
        elif attack_type == AttackType.EVOBA:
            img = strategy_objects[i].get_best_candidate()

        plt.imsave(fname, img / 255)
        mlflow.log_artifact(fname)
        os.remove(fname)

        fname = f"{i}_original.png"
        plt.imsave(fname, unperturbed_images[i] / 255)
        mlflow.log_artifact(fname)
        os.remove(fname)

    mlflow.end_run()

    return "SUCCESS"


# def get_grid_pixel_groups(patch_size, image_size):
#     # Generate grid of areas (one area = one bandit)
#     assert image_size % patch_size == 0
#
#     pixel_groups = []
#     for i in range(int(image_size / patch_size)):
#         for j in range(int(image_size / patch_size)):
#             current_group = []
#             for pixel_i in range(patch_size):
#                 for pixel_j in range(patch_size):
#                     pixel = (patch_size * i + pixel_i, patch_size * j + pixel_j)
#                     current_group.append(pixel)
#             pixel_groups.append(current_group)
#
#     return pixel_groups


def get_grid_pixel_groups(patch_size, image_size):
    """
    Generates a grid of rectangles of size patch_size to cover an image of size image_size with no padding.

    Args:
        patch_size: A tuple of two integers representing the size of the rectangle that will be used to patch the image
            and generate a rectangular grid
        image_size: A tuple of two integers representing the size of the image to be patched.

    Returns:
        A list of lists of integer pairs, each list of integer pairs representing pixel indices belonging to the same
            rectangular patch.
    """
    pixel_groups = []
    for i in range(math.ceil(image_size[0] / patch_size[0])):
        for j in range(math.ceil(image_size[1] / patch_size[1])):
            current_group = []
            for pixel_i_delta in range(patch_size[0]):
                for pixel_j_delta in range(patch_size[1]):
                    pixel_i = patch_size[0] * i + pixel_i_delta
                    pixel_j = patch_size[1] * j + pixel_j_delta
                    if pixel_i < image_size[0] and pixel_j < image_size[1]:
                        current_group.append((pixel_i, pixel_j))
            pixel_groups.append(current_group)

    return pixel_groups
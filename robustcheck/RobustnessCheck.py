import numpy as np

from robustcheck import config
from tqdm import tqdm
from robustcheck.utils import PRINT_SEPARATOR
from robustcheck.utils.metrics import image_distance


class RobustnessCheck:
    """ Main entrypoint to the package: used to run adversarial robustness benchmarks against image classifiers.

    It encapsulates the target model, a labelled dataset used for the robustness assessment, the attack to be used
    to run the robustness check, and the attack's parameters. It provides a method to run the robustness check of the
    model against the dataset by using one of the black-box adversarial attacks we provide as part of the package.

    Attributes:
        model: Target model to be assessed from a robustness point of view. This has to expose a predict method that
            returns the output probability distributions when provided a batch of images as input.
        x_test: An array of images, each of them represented as an array (HxWxC). This represents the sample of images
            that will be used for running the robustness check.
        y_test: An array of integers representing the correct class indexes of the images in x_test.
        attack: A types.AttackType enum field specifying which attack to use to run the robustness check. Most common
            choice is AttackType.EVOBA.
        attack_params: A dictionary mapping parameters that the chosen attack expects and values. In case some mandatory
            attack parameters are not specified, these will be filled automatically according to default values that
            can be found in config.DEFAULT_PARAMS.

    Methods:
        run_robustness_check(self): Runs the specified attack against the model for each image from x_test with
            corresponding label from y_test. Returns a dictionary containing the robustness stats.
        print_robustness_stats(self): Prints the robustness stats of the model against the input dataset in a
            human-readable format. This runs no computation per se, but just prints cached  robustness stats as produced
            by run_robustness_check(self). Therefore, it needs run_robustness_check(self) to have completed successfully
            before being called, otherwise it will raise an exception.
    """
    def __init__(self, model, x_test, y_test, attack, attack_params):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test

        if len(self.x_test) != len(self.y_test):
            raise Exception(
                f"Lengths of x_test and y_test do not match: {len(self.x_test)} vs {len(self.y_test)}"
            )

        self._dataset_size = len(x_test)

        if attack not in config.SUPPORTED_ATTACKS:
            raise Exception(
                f"{attack} is not one of the supported backend types: {config.SUPPORTED_ATTACKS}"
            )

        self.attack = attack
        self.attack_params = attack_params

        for attack_param_key in config.DEFAULT_PARAMS[attack]:
            if attack_param_key not in self.attack_params:
                self.attack_params[attack_param_key] = config.DEFAULT_PARAMS[attack][
                    attack_param_key
                ]

        # TODO: check if attack_params are the ones corresponding to the attack. Map these in robustcheck.config

        print("Running accuracy evaluation")

        proba_outputs = self.model.predict(x_test)
        self._y_pred = np.argmax(proba_outputs, axis=1)

        self._correct_pred_mask = self._y_pred == self.y_test
        self._accuracy = np.mean(self._correct_pred_mask)

        print("Accuracy: ", self._accuracy)

        self._index_to_adversarial_strategy = {}
        self._stats = {}

    def run_robustness_check(self):
        """
        Runs the robustness check of the model against the correctly classified images from x_test. Note there is no
        point in adversarially perturbing the images that are already misclassified.

        Returns:
            A dictionary containing statistics about the robustness of the model. These are based on the success rates,
            adversarial distances and counts of queries required until successful perturbations of the underlying
            black-box adversarial attack that is used.

        """
        attack_class = config.SUPPORTED_ATTACKS[self.attack]
        index_to_adversarial_strategy = {}

        for index in tqdm(range(self._dataset_size)):
            if self._correct_pred_mask[index]:
                img = self.x_test[index]
                label = self.y_test[index]
                index_to_adversarial_strategy[index] = attack_class(
                    model=self.model,
                    img=img,
                    label=label,
                    **self.attack_params,
                )

                no_steps = index_to_adversarial_strategy[index].run_adversarial_attack()

                assert (
                    no_steps > 0
                )  # This should hold as any correctly classified image requires at least one query

        self._index_to_adversarial_strategy = index_to_adversarial_strategy
        stats = self._compute_robustness_stats()

        self._stats = stats

        return stats

    def _compute_robustness_stats(self):
        if self._index_to_adversarial_strategy == {}:
            raise Exception(
                "There is no adversarial strategy dictionary computed as part of this instance"
            )

        successful_perturbation_count = 0
        successful_perturbation_queries = []
        successful_perturbation_l0_distances = []
        successful_perturbation_l2_distances = []
        successful_perturbation_indices = []

        failed_perturbation_count = 0
        failed_perturbation_indices = []

        adv_evo_strategy_indices = self.get_adversarial_strategy_indices()

        for i in adv_evo_strategy_indices:
            img = self._index_to_adversarial_strategy[i].img

            if self._index_to_adversarial_strategy[i].is_perturbed():
                successful_perturbation_count += 1
                successful_perturbation_queries.append(
                    self._index_to_adversarial_strategy[i].queries
                )

                curr_l0 = image_distance(
                    self._index_to_adversarial_strategy[i].get_best_candidate(),
                    img,
                    norm="L0",
                )
                successful_perturbation_l0_distances.append(curr_l0)

                curr_l2 = image_distance(
                    self._index_to_adversarial_strategy[i].get_best_candidate(),
                    img,
                    norm="L2",
                )
                successful_perturbation_l2_distances.append(curr_l2)

                successful_perturbation_indices.append(i)
            else:
                failed_perturbation_count += 1
                failed_perturbation_indices.append(i)

        img_shape = np.shape(
            self._index_to_adversarial_strategy[adv_evo_strategy_indices[0]].img
        )
        count_px = img_shape[0] * img_shape[1] * img_shape[2]

        # Will report l2 distances on [0,1] pixel scale, as this is usual in the literature
        # e.g. ImageNet is on [0,255]. Note l0 doesn't need to be normalised, as it's a count
        img_scale = self._index_to_adversarial_strategy[
            adv_evo_strategy_indices[0]
        ].pixel_space_max

        return {
            "count_succ": int(successful_perturbation_count),
            "queries_succ": successful_perturbation_queries,
            "l0_dists_succ": successful_perturbation_l0_distances,
            "l2_dists_succ": successful_perturbation_l2_distances,
            "indices_succ": successful_perturbation_indices,
            "count_fail": int(failed_perturbation_count),
            "indices_fail": failed_perturbation_indices,
            "queries_succ_mean": np.mean(successful_perturbation_queries),
            "l0_dists_succ_mean": np.mean(successful_perturbation_l0_distances),
            "l2_dists_succ_mean": np.mean(successful_perturbation_l2_distances)
            / img_scale,
            "l2_dists_succ_mean_pp": np.mean(successful_perturbation_l2_distances)
            / (count_px * img_scale),
        }

    def get_stats(self):
        """
        Returns:
             A dictionary containing the statistics of the robustness check if it was run or raises an exception if not.
        """
        if self._stats == {}:
            raise Exception("No stats have been computed as part of this instance")
        return dict.copy(self._stats)

    def get_adversarial_strategy_indices(self):
        adv_evo_strategy_indices = list(self._index_to_adversarial_strategy.keys())
        return adv_evo_strategy_indices

    def get_adversarial_strategy_perturbed_flag(self, index):
        adversarial_strategy = self._index_to_adversarial_strategy[index]
        return adversarial_strategy.is_perturbed()

    def get_adversarial_strategy_perturbed_image(self, index):
        adversarial_strategy = self._index_to_adversarial_strategy[index]
        return adversarial_strategy.get_best_candidate()

    def print_robustness_stats(self):
        """
        Prints the robustness check statistics in a human-readable format.
        """
        if self._stats == {}:
            raise Exception("No stats have been computed as part of this instance")

        count_succ = self._stats["count_succ"]
        count_fail = self._stats["count_fail"]
        count_total = count_succ + count_fail

        queries_succ_mean = self._stats["queries_succ_mean"]
        l0_dists_succ_mean = self._stats["l0_dists_succ_mean"]

        queries_succ = self._stats["queries_succ"]
        l0_dists_succ = self._stats["l0_dists_succ"]

        l2_dists_succ_mean_pp = self._stats["l2_dists_succ_mean_pp"]

        print()
        print("EvoBA STATS (L0 attack)")
        print(PRINT_SEPARATOR)

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
        print(PRINT_SEPARATOR)
        print()

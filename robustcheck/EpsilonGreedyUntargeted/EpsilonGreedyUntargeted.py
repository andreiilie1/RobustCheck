import numpy as np
import random
from robustcheck.types.UntargetedAttack import UntargetedAttack


class EpsilonGreedyUntargeted(UntargetedAttack):
    """Black-box, untargeted adversarial attack against image classifiers.

    It encapsulates the target model and image and provides a method to run the adversarial attack. The attack
    samples groups of pixels to adversarially perturb according to a classic epsilon-greedy strategy. The reward
    is represented by the decrease in the probability to be classified correctly of the target image by the target
    model. The attack samples a pixel from the group that provided the highest average reward so far with
    probability 1-epsilon, and a pixel from a random group with probability epsilon.

    Attributes:
        model: Target model to be attacked. This has to expose a predict method that returns the
            output probability distributions when provided a batch of images as input.
        img: An array (HxWxC) representing the target image to be perturbed.
        label: An integer representing the correct class index of the image.
        pixel_groups: An array of arrays of pairs of integers. Each second level array represents the indices of pixels
            that get attacked as part of the same pixel group. Usual approaches are to have these groups created based
            on objectness or on spatial proximity (e.g. in a grid-like setup).
        epsilon: A float representing the probability of exploration (choosing a random group of pixels to be perturbed)
            in the classic epsilon-greedy strategy.
        pixel_space_max: A number (integer or float) representing the maximum value pixels can take in the image space.
            This is used for extracting normalised metrics about the attack success later on.
        verbose: A boolean flag which, when set to True, enables printing info on the attack results.

    Methods:
        get_best_candidate(self): Returns the fittest individual in the active generation.
        is_perturbed(self): Returns a boolean representing whether a successful adversarial perturbation has been
            achieved in the active generation.
        run_adversarial_attack(self, steps=100): Runs the adversarial attack based on the evolutionary strategy until a
            successful adversarial perturbation was found or until steps generations were explored. Returns the total
            number of generations before the stopping condition was reached.
    """
    def __init__(self, model, img, label, pixel_groups, epsilon=0.1, pixel_space_max=1.0, steps=1000, verbose=False):
        UntargetedAttack.__init__(self, model, img, label)  # Each instance encapsulates the model and image to perturb

        self._perturbed_img = np.copy(img)  # self._perturbed_img is the variable we will iteratively perturb
        self._model_perturbed_prediction = self.model.predict(np.array([self._perturbed_img]))[0]
        self.queries = 1

        self.pixel_groups = pixel_groups
        self.number_groups = len(self.pixel_groups)

        assert 0 <= epsilon <= 1  # epsilon is a probability (of exploration), needs to be between 0 and 1
        self.epsilon = epsilon
        self.pixel_space_max = pixel_space_max
        self.steps = steps

        self.verbose = verbose

        # self._values contains the historical average rewards per pixel group
        self._values = [0.0 for _ in range(self.number_groups)]

        # self._counts contains the historical count of explorations per pixel group
        self._counts = [0 for _ in range(self.number_groups)]

    def select_group(self):
        """
        This is the core method that trades off between exploration and exploitation, as expected in classic
        epsilon-greedy strategies. Here, exploration is represented by sampling a random group of pixels, while
        exploitation means selecting a group of pixels with the highest average reward observed so far.
        """
        if random.random() > self.epsilon:
            # Pick a pixel group with the highest historical reward with probability 1 - self.epsilon.
            max_reward_indices = np.flatnonzero(np.array(self._values) == np.max(np.array(self._values)))
            max_group_index = np.random.choice(max_reward_indices)
            return max_group_index
        else:
            # Pick a random pixel group with probability self.epsilon.
            random_group_index = random.randrange(self.number_groups)
            return random_group_index

    def update(self, chosen_group, reward):
        """
        Updates a pixel group chosen_group according to an observed reward. This will update the corresponding
        group value and count of historical observation. This updates the instance fields _values and _counts.

        Args:
            chosen_group: An integer representing the index of the pixel group that will get updated after a new reward
                was observed.
            reward: The reward used to update the pixel group value and count.
        Returns:
            A float representing the updated value of the historical average reward of the chosen group.
        """
        self._counts[chosen_group] = self._counts[chosen_group] + 1

        n = self._counts[chosen_group]
        value = self._values[chosen_group]

        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self._values[chosen_group] = new_value
        return new_value

    def explore_attack_group(self, group_index):
        """
        Explores the potential reward obtained by sampling the attacked pixel from a fixed group.

        Args:
            group_index: An integer representing the index of the pixel group that the method will attempt perturbing.
        Returns:
            A dictionary containing information about the perturbation attempt. The dictionary contains the following
            fields:
                "potential_reward": A float representing the expected reward by perturbing the target group.
                "altered_image": A three-dimensional array representing the perturbed image after applying the
                    group_index group perturbation.
                "prob_before": A float representing the probability of the perturbed image to be classified correctly
                    by the target model before applying the group_index group perturbation.
                "prob_after": A float representing the probability of the perturbed image to be classified correctly
                    by the target model after applying the group_index group perturbation.
                "pred_after": An array of floats representing the probability distribution of the perturbed image as
                    output by the target model after applying the group_index group perturbation.
        """
        attack_group = self.pixel_groups[group_index]
        count_pixels_group = len(attack_group)
        attack_pixel_index = random.randrange(count_pixels_group)
        attack_pixel = attack_group[attack_pixel_index]

        candidate_next_perturbed_img = self._perturbed_img.copy()

        for ch in range(np.shape(self.img)[2]):
            value = random.randint(0, 255)
            candidate_next_perturbed_img[attack_pixel[0]][attack_pixel[1]][ch] = value

        correct_class_prob_before = self._model_perturbed_prediction[self.label]

        pred_after = self.model.predict(np.array([candidate_next_perturbed_img]))[0]
        self.queries += 1

        correct_class_prob_after = pred_after[self.label]
        potential_reward = correct_class_prob_before - correct_class_prob_after

        return {
            "potential_reward": potential_reward,
            "altered_image": candidate_next_perturbed_img,
            "prob_before": correct_class_prob_before,
            "prob_after": correct_class_prob_after,
            "pred_after": pred_after
        }

    def run_adversarial_attack(self):
        """
        Runs the adversarial attack.

        Returns:
             An integer representing the number of attack steps until either the attack was successful or the maximum
             steps threshold was reached.
        """
        trial_index = 0
        while trial_index < self.steps and not self.is_perturbed():
            attack_group = self.select_group()  # Select the target pixel group to perturb

            # Simulate attacking the target pixel group and retrieve the potential reward
            attack_result = self.explore_attack_group(attack_group)

            potential_reward = attack_result["potential_reward"]
            altered_image = attack_result["altered_image"]

            # Only update the perturbed image self._perturbed_img if the potential reward is positive
            if potential_reward > 0:
                self._perturbed_img = altered_image
                self._model_perturbed_prediction = attack_result["pred_after"]

            # Update the average historical reward of the target pixel group no matter if the reward was positive or not
            self.update(attack_group, potential_reward)

            trial_index += 1

        if self.is_perturbed() and self.verbose:
            print(f"Image successfully perturbed in {trial_index} rounds")
        elif not self.is_perturbed() and self.verbose:
            print(f"The attack did not succeed within {trial_index} rounds")

        if self.verbose:
            print("Correct label:", self.label)
            print("Predicted label:", np.argmax(self._model_perturbed_prediction))

        return trial_index

    def get_best_candidate(self):
        return self._perturbed_img

    def is_perturbed(self):
        """
        Returns:
            A boolean representing whether the adversarial attack has been successful
        """
        pred_label = np.argmax(self._model_perturbed_prediction)
        correct_output = (pred_label == self.label)
        return not correct_output

    # TODO: move to /attacks module together with EvoStrategyUniformUntargeted, keep only RobustnessCheck in main folder

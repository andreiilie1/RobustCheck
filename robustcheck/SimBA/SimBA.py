import numpy as np
import random
from robustcheck.types.UntargetedAttack import UntargetedAttack


class SimBA(UntargetedAttack):
    """ Black-box, untargeted adversarial attack against image classifiers.

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
        verbose: A boolean flag which, when set to True, enables printing info on the attack results.

    Methods:
        get_best_candidate(self): Returns the fittest individual in the active generation.
        is_perturbed(self): Returns a boolean representing whether a successful adversarial perturbation has been
            achieved in the active generation.
        run_adversarial_attack(self, steps=100): Runs the adversarial attack based on the evolutionary strategy until a
            successful adversarial perturbation was found or until steps generations were explored. Returns the total
            number of generations before the stopping condition was reached.
    """
    def __init__(
            self,
            model,
            img,
            label,
            epsilon=0.1,
            pixel_space_min=0.0,
            pixel_space_max=1.0,
            steps=1000,
            verbose=False
    ):
        UntargetedAttack.__init__(self, model, img, label)  # Each instance encapsulates the model and image to perturb

        self._perturbed_img = np.copy(img)  # self._perturbed_img is the variable we will iteratively perturb
        self._model_perturbed_prediction = self.model.predict(np.array([self._perturbed_img]))[0]
        self.queries = 1

        self.epsilon = epsilon

        self.pixel_space_min = pixel_space_min
        self.pixel_space_max = pixel_space_max

        self.steps = steps

        self.verbose = verbose

        self._unexplored_pixels = [(i, j) for i in range(np.shape(img)[0]) for j in range(np.shape(img)[1])]

    def run_adversarial_attack(self):
        """
        Runs the adversarial attack.

        Returns:
             An integer representing the number of attack steps until either the attack was successful or the maximum
             steps threshold was reached.
        """
        trial_index = 0
        while trial_index < self.steps and not self.is_perturbed() and len(self._unexplored_pixels) > 0:
            trial_index += 1

            sampled_pixel_index = random.randint(0, len(self._unexplored_pixels) - 1)
            (sampled_pixel_x, sampled_pixel_y) = self._unexplored_pixels[sampled_pixel_index]
            self._unexplored_pixels.remove((sampled_pixel_x, sampled_pixel_y))

            candidate_plus_perturbed_img = self._perturbed_img.copy()
            candidate_plus_perturbed_img[sampled_pixel_x][sampled_pixel_y][0] += self.epsilon * self.pixel_space_max
            candidate_plus_perturbed_img[sampled_pixel_x][sampled_pixel_y][0] = min(
                candidate_plus_perturbed_img[sampled_pixel_x][sampled_pixel_y][0], self.pixel_space_max
            )

            candidate_plus_distribution = self.model.predict(np.array([candidate_plus_perturbed_img]))[0]
            self.queries += 1

            if candidate_plus_distribution[self.label] < self._model_perturbed_prediction[self.label]:
                self._model_perturbed_prediction = candidate_plus_distribution
                # l0_distance += 1
                # l2_distance_sq += \
                #     (img_pos[i][j][0] - img_clean[i][j][0]) ** 2 - (img[i][j][0] - img_clean[i][j][0]) ** 2
                self._perturbed_img = candidate_plus_perturbed_img
            else:
                candidate_minus_perturbed_img = self._perturbed_img.copy()
                candidate_minus_perturbed_img[sampled_pixel_x][sampled_pixel_y][0] -= self.epsilon * self.pixel_space_max
                candidate_minus_perturbed_img[sampled_pixel_x][sampled_pixel_y][0] = max(
                    candidate_minus_perturbed_img[sampled_pixel_x][sampled_pixel_y][0], self.pixel_space_min
                )

                candidate_minus_distribution = self.model.predict(np.array([candidate_minus_perturbed_img]))[0]
                self.queries += 1

                if candidate_minus_distribution[self.label] < self._model_perturbed_prediction[self.label]:
                    # curr_probs_distribution = res_neg_distribution
                    # curr_prob = res_neg
                    # l0_distance += 1
                    # l2_distance_sq += \
                    #     (img_neg[i][j][0] - img_clean[i][j][0]) ** 2 - (img[i][j][0] - img_clean[i][j][0]) ** 2
                    # img = img_neg
                    self._model_perturbed_prediction = candidate_minus_distribution
                    self._perturbed_img = candidate_minus_perturbed_img

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

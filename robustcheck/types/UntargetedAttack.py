from abc import ABC, abstractmethod


class UntargetedAttack(ABC):
    """Abstract class for untargeted adversarial attacks

        This is an abstract class that provides a template for standard untargeted adversarial attacks.

        Attributes:
            model: Target model to be attacked. This has to expose a predict method that returns the
                output probability distributions when provided a batch of images as input.
            img: An array (HxWxC) representing the target image to be perturbed.
            label: An integer representing the correct class index of the image.

        Methods:
            run_adversarial_attack(self): Abstract method, its implementation will run the adversarial attack.
            is_perturbed(self): Abstract method, its method will return a boolean indicating whether a successful
                untargeted adversarial perturbation was found.

        """
    def __init__(self, model, img, label):
        """Inits UntargetedAttack with the target model, image to perturb, and the index of the correct image label"""
        self.model = model
        self.img = img
        self.label = label

    @abstractmethod
    def run_adversarial_attack(self):
        """The implementation of this method will run the actual adversarial attack"""
        pass

    @abstractmethod
    def is_perturbed(self):
        """The implementation of this method will return whether the adversarial attack has been successful"""
        pass

from robustcheck import config
from robustcheck.utils import get_evoba_stats, print_evoba_stats
from robustcheck.types.AttackType import AttackType


class RobustnessCheck:
    def __init__(self, model, x_test, y_test, attack, attack_params):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test

        self.attack = attack
        self.attack_params = attack_params

        if attack not in config.SUPPORTED_ATTACKS:
            raise Exception(f"{attack} is not one of the supported backend types: {config.SUPPORTED_ATTACKS}")

        # TODO: check if attack_params are the ones corresponding to the attack. Map these in robustcheck.config

    def run_robustness_check(self):
        AttackClass = config.SUPPORTED_ATTACKS[self.attack]

        attack_instance = AttackClass(self.model, self.x_test[0], self.y_test[0], **self.attack_params)
        count_queries = attack_instance.run_adversarial_attack(100)
        print(count_queries)
        stats = get_evoba_stats({0: attack_instance})
        print_evoba_stats(stats)

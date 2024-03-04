from robustcheck.types.AttackType import AttackType
from robustcheck.EvoStrategyUniformUntargeted import EvoStrategyUniformUntargeted
from robustcheck.EpsilonGreedyUntargeted import EpsilonGreedyUntargeted
from robustcheck.SimBA import SimBA


SUPPORTED_ATTACKS = {
    AttackType.EVOBA: EvoStrategyUniformUntargeted,
    AttackType.EPSGREEDY: EpsilonGreedyUntargeted,
    AttackType.SIMBA: SimBA
}

DEFAULT_PARAMS = {
    AttackType.EVOBA:
        {
            "generation_size": 30,
            "one_step_perturbation_pixel_count": 1,
            "steps": 100,
        },
    AttackType.EPSGREEDY:
        {
            "epsilon": 0.1,
            "steps": 1000,
        },
    AttackType.SIMBA:
        {
            "epsilon": 0.1,
            "steps": 100,
        }
}

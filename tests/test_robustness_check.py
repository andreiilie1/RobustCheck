import numpy as np
import pytest
import sys
import os
from robustcheck import RobustnessCheck
from robustcheck.types.AttackType import AttackType


def test_robustness_check():
    CIFAR100_VGG_PATH = os.path.join(os.path.dirname(__file__), "resources", "models", "vgg")
    sys.path.append(CIFAR100_VGG_PATH)

    from cifar100vgg import CIFAR100VGG
    model = CIFAR100VGG(train=False)

    x_test = np.load(os.path.join(os.path.dirname(__file__), "resources", "data", "cifar100_sample_x.npy"))
    y_test = np.load(os.path.join(os.path.dirname(__file__), "resources", "data", "cifar100_sample_y.npy"))

    x_test = x_test.astype('float32')
    x_test = x_test.astype('int')

    preds = model.predict(x_test)
    true_labels = np.argmax(y_test, axis=1)

    correct_count = np.sum(true_labels == np.argmax(preds, axis=1))

    rc = RobustnessCheck(
        model=model,
        x_test=x_test,
        y_test=true_labels,
        attack=AttackType.EVOBA,
        attack_params={
            "generation_size": 30,
            "one_step_perturbation_pixel_count": 1,
            "pixel_space_int_flag": True,
            "pixel_space_min": 0,
            "pixel_space_max": 255,
            "verbose": False
        }
    )

    rc.run_robustness_check()
    rc.print_robustness_stats()

    stats = rc.get_stats()

    assert len(rc.get_adversarial_strategy_indices()) == correct_count
    assert stats["count_succ"] > int(0.75 * correct_count)
    assert stats["queries_succ_mean"] < 500
    assert stats["l0_dists_succ_mean"] < 40

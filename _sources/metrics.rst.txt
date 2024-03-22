.. _robustness_metrics:

Robustness metrics
==================

We define below the robustness metrics that our package assesses:

* ``count_succ``: how many image samples were successfully perturbed,
* ``count_fail``: how many image samples were not successfully perturbed,
* ``queries_succ_mean``: the average count of black-box model queries required to adversarially perturb the successfully
  perturbed samples,
* ``l0_dists_succ_mean``: the average L0 distance (count of pixels, where each channel is counted separately) of the
  adversarial perturbations added to the successfully perturbed samples,
* ``l2_dists_succ_mean``: the average L2 distance (Euclidean norm) of the adversarial perturbations added to the
  successfully perturbed samples. This is computed relatively to the pixel scale. For example, if the image pixels have
  values in the range [0, 255], the L2 distance will be normalised by 255,
* ``l2_dists_succ_mean_pp``: the average L2 distance (Euclidean norm) of the adversarial perturbations added to the
  successfully perturbed samples. This is computed relatively to both the pixel scale and count of pixels. This can be
  interpreted as the average relative per-pixel perturbation.

These are all part of the dictionary that ``rc.run_robustness_check()`` returns. They are also found in the artifacts
logged by ``save_robustness_stats_artifacts(rc, path_to_output)`` and in the MLFlow logs produced by
``generate_mlflow_logs(rc, run_name, experiment_name, tracking_uri)``. For details on how to use these methods see
:ref:`how to review the robustness metrics <review_robustness_metrics>`

The dictionary returned by ``rc.run_robustness_check()`` and saved as an artifact by
``save_robustness_stats_artifacts(rc, path_to_output)`` contains the raw robustness check results as well:

* ``indices_succ``: a list containing the indices of the images that were successfully perturbed,
* ``indices_fail``: a list containing the indices of the images that were successfully perturbed,
* ``l0_dists_succ``: a list containing the L0 distances of the images that were successfully perturbed,
* ``l2_dists_succ``: a list containing the L2 distances of the images that were successfully perturbed (with no
  normalisation),
* ``queries_succ``: a list containing the black-box model query counts required to perturb the images that were
  successfully perturbed.

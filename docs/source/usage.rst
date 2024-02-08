Usage
=====

.. _installation:

Installation
------------

To use RobustCheck, first install it using pip:

.. code-block:: console

   (.venv) $ pip install robustcheck

Data and model preparation
--------------------------

Ensure your model exposes a ``.predict(image_batch)`` function which can take as
input an ``np.array`` batch of images and produces as output an array of arrays
representing the probability distributions for each image in the batch to be
classified as any of the existing classes.

You will have to provide a sample of images (``x_test``) with their correct labels
(``test_labels``). These have to be of ``np.array`` type.

Set up a RobustnessCheck object
-------------------------------
Create a ``RobustnessCheck`` object with your model, test data, and attack parameters:

.. code-block:: python

   import robustcheck

   rc = robustcheck.RobustnessCheck(
       model=model,
       x_test=x_test,
       y_test=test_labels,
       attack=AttackType.EVOBA,
       attack_params={
           "generation_size": 160,
           "one_step_perturbation_pixel_count": 10,
           "pixel_space_int_flag": False,
           "pixel_space_min": 0,
           "pixel_space_max": 255,
           "verbose": False,
           "steps": 100,
       }
   )

Run the robustness check
------------------------
Execute the robustness check:

.. code-block:: python

   robustness_metrics = rc.run_robustness_check()

This is where you actually run the robustness checks by triggering an adversarial attack
against each image that is correctly classified by model in the provided ``x_test`` sample.

``robustness_metrics`` is a dictionary containing a mapping between robustness metrics such as
``count_succ`` (how many samples were successfully perturbed) and their values. It also contains
raw results, for example through the mapping between ``l0_dists_succ`` and a list of all L0 norms
of all successful adversarial perturbations.

Review the robustness metrics
-----------------------------
While ``robustness_metrics`` contains all relevant metrics of the robustness check, we provide
friendlier ways to review these.

There are various ways to interact with the robustness metrics that ``run_robustness_check()``
produce. You can print them in a human-readable form, generate and store artifacts containing
metrics and various plots on the disk, or generate MLFlow logs containing metrics and artifacts.

Printing robustness metrics in a human-readable form
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can print the robustness metrics to the standard output by running:

.. code-block:: python

   rc.print_robustness_stats()

This will produce an output containing all relevant metrics. For example, the output can look like:

.. code-block:: console

   EvoBA STATS (L0 attack)
   ___________________
   Perturbed successfully 13/13 images
   Average query count: 264.0769230769231
   Average l0 distance: 26.076923076923077
   Average l2 distance per pixel: 0.0006845784778314443

   Median query count: 211.0
   Median l0 dist: 21.0

   Max query count: 751
   Max l0 dist: 75
   ___________________



Generating and storing artifacts on the disk
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Another option is saving the robustness metrics and other relevant artifacts such as image-level
histograms of the relevant metrics by running the snippet below:

.. code-block:: python

   from robustcheck.utils import save_robustness_stats_artifacts
   save_robustness_stats_artifacts(rc, path_to_output)


This will produce the following artifacts at the path ``path_to_output``:

* ``l0_dists_histogram.png`` and ``l2_dists_histogram.png`` - histograms of the successful adversarial perturbation norms
* ``queries_histogram.png`` - a histogram of the query counts needed for successful adversarial perturbations
* ``robustness_stats.json`` - a JSON file containing both the relevant robustness metrics and the raw results
  (non-aggregated lists of query counts and perturbation norms).
Generating MLFlow logs
^^^^^^^^^^^^^^^^^^^^^^
Finally, you can use MLFlow to generate logs for the robustness check. These will contain all metrics and artifacts
of the methods above, but will additionally use MLFlow's UI to visualise both the perturbed and unperturbed images.
This should provide you a qualitative understanding of how successful perturbations look like and assess how perceptible
these are. You can generate MLFlow logs by running:

.. code-block:: python

   from robustcheck.utils import save_robustness_stats_artifacts
   generate_mlflow_logs(rc, run_name, experiment_name, tracking_uri)

This will generate MLFlow compatible artifacts under the run ``run_name`` and under the experiment ``experiment_name``
stored at the ``tracking_uri`` location, which can either be a local path or a dedicated MLFlow server. Read more
about how to use MLFlow `here <https://mlflow.org/docs/latest/getting-started/index.html>`_.

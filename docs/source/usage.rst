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

   _ = rc.run_robustness_check()

This is where we actually run the robustness checks by triggering an adversarial attack
against each image that is correctly classified by model in the provided ``x_test`` sample.

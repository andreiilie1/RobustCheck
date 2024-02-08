.. RobustCheck documentation master file, created by
   sphinx-quickstart on Sat Dec 23 13:21:15 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to RobustCheck's documentation!
=======================================
RobustnessCheck is a Python package designed for evaluating the robustness of image
classification machine learning models. It provides tooling to apply simple, yet effective
and efficient black-box untargeted adversarial attacks against models that expose a batch
predict function that outputs probability distributions.

This tool is essential for researchers and practitioners who wish to assess the
resilience of their models to adversarial perturbations or their robustness in a
more general way.

The black-box, untargeted adversarial attacks used for robustness assessment are EvoBA,
which is a peer-reviewed and published approach [1], and EpsGreedy, which is a variation of
EvoBA following the classic Epsilon-Greedy exploration strategy.

Check out the :doc:`usage` section for further information, including how to
:ref:`install <installation>` the project.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   metrics
   RobustnessCheck
   EvoStrategyUniformUntargeted
   EvoStrategy



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Bibliography
============
[1] Neural Information Processing: 28th International Conference, ICONIP 2021, Sanur,
Bali, Indonesia, December 8–12, 2021, Proceedings, Part III
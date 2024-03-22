.. RobustCheck documentation master file, created by
   sphinx-quickstart on Sat Dec 23 13:21:15 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

RobustCheck documentation
=========================

.. image:: ../../assets/RobustnessCheck-logo2.png
   :height: 400px

RobustnessCheck is a Python package designed for evaluating the robustness of image
classification machine learning models. It provides tooling to apply simple, yet effective
and efficient black-box untargeted adversarial attacks against models that expose a batch
predict function that outputs probability distributions.

* Check out the :doc:`usage` section for getting started, including how to
  :ref:`install <installation>` the package.

* :doc:`metrics` introduces the main robustness metrics that RobustCheck generates.

* :doc:`RobustnessCheck` provides the documentation for the main functionality of our
  package.

* :doc:`dump_metrics` provides the documentation for the functions used to save the
  robustness assessment results.

.. toctree::
   :maxdepth: 2
   :caption: Main content:

   usage
   metrics
   RobustnessCheck
   dump_metrics

The black-box, untargeted adversarial attacks used for robustness assessment are EvoBA,
which is a peer-reviewed and published approach [1], and EpsGreedy, which is a variation of
EvoBA following the classic Epsilon-Greedy exploration strategy.

We provide the documentation for the underlying adversarial attacks that RobustCheck uses to
generate the robustness metrics.

.. toctree::
   :maxdepth: 1
   :caption: Adversarial attacks documentation:

   EvoStrategyUniformUntargeted
   EpsilonGreedyUntargeted
   UntargetedAttack
   EvoStrategy

``EvoStrategyUniformUntargeted`` and ``EpsilonGreedyUntargeted`` are the adversarial attacks
used by RobustCheck. We provide an abstract class ``UntargetedAttack`` that both of the attacks
implement. Developers interested to extend RobustCheck via the addition of other relevant
adversarial attacks are encouraged to do it by providing new implementations of
``UntargetedAttack``.

We further provide ``EvoStrategy``, a generic abstract class for evolutionary search strategies.
``EvoStrategyUniformUntargeted`` implements both this and ``UntargetedAttack``. We believe that
there are many other evolutionary strategies that would fulfill the efficiency and effectiveness
criteria required by the RobustCheck package, therefore we encourage potential contributors to
consider adding attacks that implement both ``EvoStrategy`` and ``UntargetedAttack``.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Bibliography
============
[1] Neural Information Processing: 28th International Conference, ICONIP 2021, Sanur,
Bali, Indonesia, December 8–12, 2021, Proceedings, Part III
# RobustnessCheck Package

<img src="assets/RobustnessCheck-logo.png" width="400" />

"<i>'As the oak resists the wind and grows stronger, let our models 
face adversarial challenges and emerge more robust.'</i> - Marcus Aurelius" - ChatGPT

## Overview
RobustnessCheck is a Python package designed for evaluating the robustness of image classification machine learning 
models. It provides tooling to apply simple, yet effective and efficient black-box untargeted adversarial attacks 
against models that expose a batch predict function that outputs probability distributions. 

This tool is essential for researchers and practitioners who wish to assess the resilience of their models to 
adversarial perturbations or their robustness in a more general way.

The black-box, untargeted adversarial attacks used for robustness assessment are **EvoBA**, which is a peer-reviewed
and published approach [1], and **EpsGreedy**, which is a variation of **EvoBA** following the classic 
Epsilon-Greedy exploration strategy.

## Documentation
An HTML documentation of the package is available <b> <a href="https://andreiilie1.github.io/RobustCheck/index.html#">here</a> </b>.

<i> This was generated using <a href="https://www.sphinx-doc.org">Sphinx</a>. </i>
## Features
- **Black-box Untargeted Adversarial Attack**: Evaluate model performance under adversarial conditions without the need 
for internal model details. 
- **Customizable Attack Parameters**: Fine-tune the attack to suit specific needs and scenarios.
- **Easy to use**: Users are able to run robustness checks with very few and general lines of code that we provide 
documentation for. We also provide default settings and thumb-rules for configuring the robustness checks such that 
users do not  need to spend time on fine-tuning the attacks if they are only interested in a general set of 
robustness metrics.
- **Simple Integration**: Designed to work with models that have a .predict function, facilitating easy integration 
with existing machine learning workflows. Being a black-box adversarial attack, this can be integrated with a wide
variety of models, so not only with deep neural networks. 

## Installation
To install RobustnessCheck, clone this repo and run the following command in the base folder:

```
python -m pip install .
```

## Usage
To use RobustnessCheck, follow these steps:

### Step 1: Import the package and prepare your model
Ensure your model exposes a `.predict(image_batch)` function which can take as input an `np.array` batch of
images and produces as output an array of arrays representing the probability distributions for each image in the 
batch to be classified as any of the existing classes. 

Import the RobustnessCheck package as follows:
```
import robustcheck
```

### Step 2: Prepare Your Data

Provide a sample of images (`x_test`) with their correct labels (`test_labels`). These have to be of `np.array` type.

### Step 3: Configure and Run Robustness Check

Create a `RobustnessCheck` object with your model, test data, and attack parameters:

```
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
```

Check the documentation to understand how to select these parameters. There are two attacks that we support so far 
that come with different tradeoffs: `AttackType.EVOBA` and `AttackType.EPSGREEDY`, each coming with a different set of
`attack_params` that can be customised. First time users are advised to use `AttackType.EVOBA`. 

### Step 4: Run Robustness Check

Execute the robustness check:

```
_ = rc.run_robustness_check()
```

This is where we actually run the robustness checks by triggering an adversarial attack against each image that is
correctly classified by `model` in the provided `x_test` sample.

### Step 5: Review Results

Print the robustness statistics:
```
rc.print_robustness_stats()
```

This will produce an output that looks like
**TODO add sample output after refactoring**

### Step 6: Interpret Results
 **TODO add what metrics mean, how to compare multiple models head-to-head, etc**

## Contributing
Contributions to RobustnessCheck are welcome. Please read our contributing guidelines (TODO: assemble these) before submitting a pull request.

## License
RobustnessCheck is licensed under MIT License.

## Support
For support, please open an issue on our GitHub issue tracker. 

For bugs, provide information on what was the expected behaviour, what was the actual behaviour, and how to reproduce 
your case.

For feature requests, explain what is the feature you would like us to add and what would be some of its use cases.

## Bibliography
[1] Neural Information Processing: 28th International Conference, ICONIP 2021, Sanur, Bali, Indonesia, December 8–12, 2021, Proceedings, Part III

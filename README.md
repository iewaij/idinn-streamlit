# idinn: Inventory-Dynamics Control with Neural Networks

`idinn` implements inventory dynamics–informed neural networks designed for solving both single-sourcing and dual-sourcing problems. Neural network controllers and inventory dynamics are implemented into easily customizable classes to enable users to find the optimal controllers for the user-specified inventory systems.

## Requirements

The basic usage of `idinn` requires a working `Python` and `PyTorch` installation. If plotting simulation result of a controller is needed, `matplotlib` should also be installed.

## Installation

The package can be installed form the git repository. To do that, run

```
python -m pip install git+https://gitlab.com/ComputationalScience/inventory-optimization.git@main
```

Or, if you want to inspect the source code and edit locally, run

```
git clone https://gitlab.com/ComputationalScience/inventory-optimization.git
cd inventory-optimization
python -m pip install -e .
```

## Example Usage

```python
import torch
from idinn.sourcing_model import SingleSourcingModel
from idinn.controller import SingleFullyConnectedNeuralController

# Initialize the sourcing model and the neural controller
sourcing_model = SingleSourcingModel(
    lead_time=0, holding_cost=5, shortage_cost=495, batch_size=32, init_inventory=10
)
controller = SingleFullyConnectedNeuralController(
    hidden_layers=[2], activation=torch.nn.CELU(alpha=1)
)
# Train the neural controller
controller.train(
    sourcing_model=sourcing_model,
    sourcing_periods=50,
    validation_sourcing_periods=1000,
    epochs=5000,
    tensorboard_writer=torch.utils.tensorboard.SummaryWriter(),
    seed=1,
)
# Simulate and plot the results
controller.plot(sourcing_model=sourcing_model, sourcing_periods=100)
# Calculate the optimal order quantity for applications
controller.forward(
    current_inventory=torch.tensor([[10]]),
    past_orders=torch.tensor([[1, 5]]),
)
```

## Documentaion

For more detailed guides and documentaion, please refer to our [documentation](https://inventory-optimization.readthedocs.io/en/latest/).

## Papers using `idinn`

We will add papers that use `ìdinn` to this list as they appear online.

* Böttcher, Lucas, Thomas Asikis, and Ioannis Fragkos. "Control of Dual-Sourcing Inventory Systems Using Recurrent Neural Networks." [INFORMS Journal on Computing](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.2022.0136) 35.6 (2023): 1308-1328.

## Contributors

* [Jiawei Li](https://gitlab.com/iewaij)
* [Thomas Asikis](https://gitlab.com/asikist)
* [Ioannis Fragkos](https://gitlab.com/ioannis.fragkos1)
* [Lucas Böttcher](https://gitlab.com/lucasboettcher)

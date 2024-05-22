# idinn: Inventory-Dynamics Control with Neural Networks

[<img src="https://gitlab.com/ComputationalScience/idinn/-/raw/main/docs/_static/youtube.png" align="center" width="60%" size="auto" alt="youtube">](https://www.youtube.com/watch?v=hUBfTWV6tWQ)

`idinn` implements **i**nventory **d**ynamics–**i**nformed **n**eural **n**etworks for solving single-sourcing and dual-sourcing problems. Neural network controllers and inventory dynamics are implemented into customizable objects with PyTorch backend to enable users to find the optimal neural controllers for the user-specified inventory systems.

## Installation

The package can be installed form PyPI. To do that, run

```
pip install idinn
```

Or, if you want to inspect the source code and edit locally, run

```
git clone https://gitlab.com/ComputationalScience/idinn.git
cd idinn
pip install -e .
```

## Example Usage

```python
import torch
from idinn.sourcing_model import SingleSourcingModel
from idinn.controller import SingleSourcingNeuralController
from idinn.demand import UniformDemand

# Initialize the sourcing model and the neural controller
sourcing_model = SingleSourcingModel(
    lead_time=0,
    holding_cost=5,
    shortage_cost=495,
    batch_size=32,
    init_inventory=10,
    demand_generator=UniformDemand(low=1, high=4),
)
controller = SingleSourcingNeuralController(
    hidden_layers=[2],
    activation=torch.nn.CELU(alpha=1)
)
# Train the neural controller
controller.fit(
    sourcing_model=sourcing_model,
    sourcing_periods=50,
    validation_sourcing_periods=1000,
    epochs=5000,
    seed=1,
)
# Simulate and plot the results
controller.plot(sourcing_model=sourcing_model, sourcing_periods=100)
# Calculate the optimal order quantity for applications
controller.forward(current_inventory=10, past_orders=[1, 5])
```

## Documentation

For tutorials and documentation, please refer to our [documentation](https://inventory-optimization.readthedocs.io/en/latest/).

## Papers using `idinn`

* Böttcher, Lucas, Thomas Asikis, and Ioannis Fragkos. "Control of Dual-Sourcing Inventory Systems Using Recurrent Neural Networks." [INFORMS Journal on Computing](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.2022.0136) 35.6 (2023): 1308-1328.

## Contributors

* [Jiawei Li](https://github.com/iewaij)
* [Thomas Asikis](https://gitlab.com/asikist)
* [Ioannis Fragkos](https://gitlab.com/ioannis.fragkos1)
* [Lucas Böttcher](https://gitlab.com/lucasboettcher)

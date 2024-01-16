# idinn: A Python Package for Inventory-Dynamics Control with Neural Networks

We provide a Python package, `idinn`, implementing inventory dynamics–informed neural 
networks designed for controlling both single-sourcing and dual-sourcing problems. 
Neural network controllers and inventory dynamics are implemented in two easily customizable 
classes, enabling users to control extensions of the provided inventory management 
systems by tailoring the implementations to their needs.

## Install

The package can be installed and edited locally by running

```
git clone https://gitlab.com/ComputationalScience/inventory-optimization.git
cd inventory-optimization
python -m pip install -e .
```

Or, if you want to install the latest version without editing the source code, run

```
python -m pip install git+https://gitlab.com/ComputationalScience/inventory-optimization.git@main
```

## Quick Start

```python
import torch
from idinn.sourcing_model import SingleSourcingModel, DualSourcingModel
from idinn.controller import (
    SingleFullyConnectedNeuralController,
    DualFullyConnectedNeuralController,
    CappedDualIndexController,
)

# Single sourcing example
sourcing_model = SingleSourcingModel(
    lead_time=0, holding_cost=5, shortage_cost=495, batch_size=32, init_inventory=10
)
controller = SingleFullyConnectedNeuralController(
    hidden_layers=[2], activation=torch.nn.CELU(alpha=1)
)
controller.train(
    sourcing_model=sourcing_model,
    sourcing_periods=50,
    validation_sourcing_periods=1000,
    epochs=5000,
    tensorboard_writer=torch.utils.tensorboard.SummaryWriter(),
    seed=1,
)
controller.plot(sourcing_model=sourcing_model, sourcing_periods=100)

# Dual sourcing example with neural controller
controller = DualFullyConnectedNeuralController(
    hidden_layers=[128, 64, 32, 16, 8, 4],
    activation=torch.nn.CELU(alpha=1),
    compressed=False,
)
dual_sourcing_model = DualSourcingModel(
    regular_lead_time=2,
    expedited_lead_time=0,
    regular_order_cost=0,
    expedited_order_cost=20,
    holding_cost=5,
    shortage_cost=495,
    batch_size=256,
    init_inventory=6,
)
controller.train(
    sourcing_model=dual_sourcing_model,
    sourcing_periods=100,
    validation_sourcing_periods=1000,
    epochs=2000,
    tensorboard_writer=torch.utils.tensorboard.SummaryWriter(),
    seed=4,
)
controller.plot(sourcing_model=sourcing_model, sourcing_periods=100)

# Dual sourcing example with capped dual index controller
controller = CappedDualIndexController()
dual_sourcing_model = DualSourcingModel(
    regular_lead_time=2,
    expedited_lead_time=0,
    regular_order_cost=0,
    expedited_order_cost=20,
    holding_cost=5,
    shortage_cost=495,
    batch_size=1,
    init_inventory=6,
)
controller.train(sourcing_model=dual_sourcing_model, sourcing_periods=100)
controller.plot(sourcing_model=dual_sourcing_model, sourcing_periods=100)
```
idinn: Inventory-Dynamics Control with Neural Networks
======================================================

We provide a Python package, `idinn`, implementing inventory dynamics–informed neural 
networks designed for controlling both single-sourcing and dual-sourcing problems. 
Neural network controllers and inventory dynamics are implemented in two easily customizable 
classes, enabling users to control extensions of the provided inventory management 
systems by tailoring the implementations to their needs.

Installation
------------

The package can be installed directly form Gitlab repository. To do that, run

.. code-block::
   python -m pip install git+https://gitlab.com/ComputationalScience/inventory-optimization.git@main


Or, if you want to access the source code and perform local editing, run

.. code-block::
   git clone https://gitlab.com/ComputationalScience/inventory-optimization.git
   cd inventory-optimization
   python -m pip install -e .

Quick Start
-----------

.. code-block:: python
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

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

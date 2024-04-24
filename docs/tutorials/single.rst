Solve Single-Sourcing Problems Using Neural Networks
====================================================

Single-sourcing problems are inventory management problems where only one delivery option exists. The overall objective in single-sourcing and related inventory management problems is for companies to identify the optimal order quantities to minimize costs given stochastic demand. This problem can be addressed using `idinn`. As illustrated in the :doc:`/get_started/get_started` section, we first initialize the sourcing model and its associated neural network controller. Subsequently, we train the neural network controller using data generated from the sourcing model. Finally, we can use the trained neural network controller to compute optimal order quantities.

Initialization
--------------

Since we deal with the single-sourcing problem, we use the `SingleSourcingModel` class to initialize the sourcing model. In this tutorial, let us pick a single sourcing model which has a lead time of 0 (i.e., the order arrives immediately after it is placed) and an initial inventory of 10. The holding cost, :math:`h`, and the shortage cost, :math:`b`, are 5 and 495, respectively. The demand is generated from a discrete uniform distribution with support :math:`[1, 4]`. Notice that both the `demand_low` and `demand_low` parameters are inclusive. In this example, we use a batch size of 32.

In `idinn`, the sourcing model is initialized as follows.

.. code-block:: python
    
   import torch
   from idinn.sourcing_model import SingleSourcingModel
   from idinn.controller import SingleSourcingNeuralController

   single_sourcing_model = SingleSourcingModel(
      lead_time=0,
      holding_cost=5,
      shortage_cost=495,
      batch_size=32,
      init_inventory=10,
      demand_distribution="uniform",
      demand_low=1,
      demand_high=4
   )

The cost at period :math:`t`, :math:`c_t`, is

.. math::

   c_t = h \max(0, I_t) + b \max(0, - I_t)\,,

where :math:`I_t` is the inventory level at the end of period :math:`t`. The higher the holding cost, the more costly it is to keep the inventory (when the inventory level is positive). The higher the shortage cost, the more costly it is to run out of stock (when the inventory level is negative). The cost can be calculated using the `get_total_cost` method of the sourcing model.

.. code-block:: python
    
   single_sourcing_model.get_total_cost()

In our example, this function should return 50 for each sample since the initial inventory is 10 and the holding cost is 5. We have 32 samples in this case, as we specified a batch size of 32.

For single-sourcing problems, we initialize the neural network controller using the `SingleSourcingNeuralController` class. In this tutorial, we use a simple neural network with 1 hidden layer and 2 neurons. The activation function is `torch.nn.CELU(alpha=1)`. The neural network controller is initialized as follows.

.. code-block:: python

    single_controller = SingleSourcingNeuralController(
        hidden_layers=[2], activation=torch.nn.CELU(alpha=1)
    )

Training
--------

Although the neural network controller has not been trained yet, we can still compute the total cost associated with its order policy. To do so, we integrate it with our previously specified sourcing model and run simulations for 100 periods.

.. code-block:: python
    
    single_controller.get_total_cost(sourcing_model=single_sourcing_model, sourcing_periods=100)

Unsurprisingly, the performance is poor because we are only using the untrained neural network in which the weights are just (pseudo) random numbers. We can train the neural network controller using the `train` method, in which the training data is generated from the given sourcing model. To better monitor the training process, we specify the `tensorboard_writer` parameter to log both the training loss and validation loss. For reproducibility, we also specify the seed of the underlying random number generator using the `seed` parameter.

.. code-block:: python

    from torch.utils.tensorboard import SummaryWriter

    single_controller.train(
        sourcing_model=single_sourcing_model,
        sourcing_periods=50,
        validation_sourcing_periods=1000,
        epochs=5000,
        seed=1,
        tensorboard_writer=SummaryWriter()
    )

After training, we can use the trained neural network controller to calculate the total cost for 100 periods with our previously specified sourcing model. The total cost should be significantly lower than the cost associated with the untrained model.

.. code-block:: python

    single_controller.get_total_cost(sourcing_model=single_sourcing_model, sourcing_periods=100)

Simulation, Plotting, and Order Calculation
------------------------------------------

We can also inspect how the controller performs in the specified sourcing environment by (i) plotting the inventory and order histories, and (ii) calculating optimal orders.

.. code-block:: python

    # Simulate and plot the results
    single_controller.plot(sourcing_model=single_sourcing_model, sourcing_periods=100)
    # Calculate the optimal order quantity for applications
    single_controller.forward(current_inventory=10, past_orders=[1, 5])

Save and Load the Model
-----------------------

It is also a good idea to save the trained neural network controller for future use. This can be done using the `save` method. The `load` method allows one to load a previously saved controller.

.. code-block:: python

    # Save the model
    single_controller.save("optimal_single_sourcing_controller.pt")
    # Load the model
    single_controller_loaded = SingleSourcingNeuralController(
        hidden_layers=[2], activation=torch.nn.CELU(alpha=1)
    )
    single_controller_loaded.load("optimal_single_sourcing_controller.pt")
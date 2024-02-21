Solve Single-Sourcing Problems Using Neural Networks
====================================================

Single-sourcing problems refers to the inventory management problem where there is only one delivery option. The company aims to determine the optimal order quantity to minimize costs. We can solve this problem with `idinn`. As demonstrated in :doc:`/get_started/get_started`, we first initialize the sourcing model and the neural network controller. Then we train the neural network controller using the training data generated from the sourcing model. Finally, we can use the trained neural network controller to compute the optimal order quantity.

Initialization
--------------

Since we deal with the single-sourcing problem, we use the `SingleSourcingModel` class to initialize the sourcing model. In this tutorial, let us pick a single sourcing model which has lead time of 0, i.e. the order arrives immediately after it is placed, the initial inventory of 10 and batch size of 32. The holding cost, :math:`h`, is 5 and the shortage cost, :math:`s`, is 495. The demand is generated from a uniform distribution with interval :math:`[0, 5)`. Note that the `high`` parameter is exclusive (open bracket), the generated demand will therefore never exceed 4. In code, the sourcing model is initialized as follows:

.. code-block:: python
    
   import torch
   from idinn.sourcing_model import SingleSourcingModel
   from idinn.controller import SingleFullyConnectedNeuralController

   single_sourcing_model = SingleSourcingModel(
      lead_time=0,
      holding_cost=5,
      shortage_cost=495,
      batch_size=32,
      init_inventory=10,
      demand_generator=torch.distributions.Uniform(low=0, high=5)
   )

The cost at period :math:`t`, :math:`c_t`, is calculated as:

.. math::

   c_t = h \cdot \max(0, I_t) + s \cdot \max(0, - I_t)

where :math:`I_t` is the inventory level at period :math:`t`. The higher the holding cost, the more costly it is to keep the inventory (when the inventory level is positive). The higher the shortage cost, the more costly it is to run out of stock (when the inventory level is negative). The cost can be calculated using the `get_cost` method of the sourcing model.

.. code-block:: python
    
   single_sourcing_model.get_cost()

which should return 50 for each sample since the initial inventory is 10 and the holding cost of 5. We have 32 samples in this case because we specified batch size at 32.

We then initialize the neural network controller for single-sourcing problems using the `SingleFullyConnectedNeuralController` class. In this tutorial, we use a simple neural network with 1 hidden layer with 2 neurons. The activation function is `torch.nn.CELU(alpha=1)`. The neural network controller is initialized as follows:

.. code-block:: python

    single_controller = SingleFullyConnectedNeuralController(
        hidden_layers=[2], activation=torch.nn.CELU(alpha=1)
    )

Training
--------

Even though the neural network controller is not trained yet, we can already use it to calculate the total cost if we use this controller for 100 periods with our previously specified sourcing model.

.. code-block:: python
    
    single_controller.get_total_cost(sourcing_model=single_sourcing_model, sourcing_periods=100)

Unsurprisingly, the performance is poor because we are only using the untrained neural network in which the weights are just random numbers. We can train the neural network controller using the `train` method where the training data is generated from the given sourcing model. To better monitor the training process, we specify the `tensorboard_writer` parameter to log the training loss and validation loss. For reproducibility, we also specify the random seed using the `seed` parameter.

.. code-block:: python

    from torch.utils.tensorboard import SummaryWriter

    single_controller.train(
        sourcing_model=sourcing_model,
        sourcing_periods=50,
        validation_sourcing_periods=1000,
        epochs=5000,
        seed=1,
        tensorboard_writer=SummaryWriter()
    )

After training, we can use the trained neural network controller to calculate the total cost for 100 periods with our previously specified sourcing model. The total cost should be significantly lower than the previous one.

.. code-block:: python

    single_controller.get_total_cost(sourcing_model=single_sourcing_model, sourcing_periods=100)

Simulation, Plotting and Order Calculation
------------------------------------------

We can also inspect how the controller perform in the specified sourcing environment by plotting the inventory and order history, and calculate optimal orders for applications.

.. code-block:: python

    # Simulate and plot the results
    single_controller.plot(sourcing_model=single_sourcing_model, sourcing_periods=100)
    # Calculate the optimal order quantity for applications
    single_controller.forward(
        current_inventory=torch.tensor([[10]]),
        past_orders=torch.tensor([[1, 5]]),
    )

Save and Load the Model
-----------------------

It is also a good idea to save the trained neural network controller for future use. This can be done using the `save` method and the `load` method.

.. code-block:: python

    # Save the model
    single_controller.save("optimal_single_sourcing_controller.pt")
    # Load the model
    single_controller_loaded = SingleFullyConnectedNeuralController(
        hidden_layers=[2], activation=torch.nn.CELU(alpha=1)
    )
    single_controller_loaded.load("optimal_single_sourcing_controller.pt")
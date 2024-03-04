Solve Dual-Sourcing Problems Using Neural Networks
==================================================

Dual-sourcing problems are similar to single-sourcing problems but are more intricate. In a dual-sourcing problem, a company has two potential suppliers for a product, each offering varying lead times (the duration for orders to arrive) and order costs (the expense of placing an order). The challenge lies in the company's decision-making process: determining which supplier to engage for each product to minimize costs given stochastic demand. We can solve dual-sourcing problems with `idinn` in a way similar to the approaches describes described in :doc:`/get_started/get_started` and :doc:`/tutorials/single`.

Initialization
--------------

To address dual-sourcing problems, we employ two main classes: `DualSourcingModel` and `DualFullyConnectedNeuralController`, responsible for setting up the sourcing model and its corresponding controller. In this tutorial, we adopt a dual-sourcing model with specific parameters: regular order lead time and expedited order lead time both set to 0, regular order cost, :math:`c^r`, at 0, expedited order cost, :math:`c^e`, at 20, initial inventory of 6, and a batch size of 256. Additionally, the holding cost, :math:`h`, is 5, while the shortage cost, :math:`s`, is 495. Demand is generated from a uniform distribution with the interval :math:`[0, 5)`. Notice that the `high` parameter is exclusive. Hence, the generated demand never exceeds 4. In our code, the sourcing model is initialized as follows.

.. code-block:: python
    
   import torch
   from idinn.sourcing_model import DualSourcingModel
   from idinn.controller import DualFullyConnectedNeuralController

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

The cost at period :math:`t`, :math:`c_t`, is

.. math::

   c_t = c^r q^r_t + c^e q^e_t + h \cdot \max(0, I_t) + s \cdot \max(0, - I_t)\,,

where :math:`I_t` is the inventory level at period :math:`t`, :math:`q^r_t` is the regular order sent at period :math:`t`, :math:`q^e_t` is the expedited order sent at period :math:`t`. The higher the holding cost, the more costly it is to keep the inventory (when the inventory level is positive). The higher the shortage cost, the more costly it is to run out of stock (when the inventory level is negative). The higher the regular or expedited order costs, the more costly it is to send the respective orders. The cost can be calculated using the `get_cost` method of the sourcing model.

.. code-block:: python
    
   dual_sourcing_model.get_cost(regular_q=0, expedited_q=0)

In our example, this function should return 30 for each sample since the initial inventory is 6, the holding cost is 5, and there is neither a regular nor expedited order. We have 256 samples in this case, as we specified a batch size of 256.

For dual-sourcing problems, we initialize the neural network controller using the `DualFullyConnectedNeuralController` class. In this tutorial, we use a simple neural network with 6 hidden layers and 128, 64, 32, 16, 8, 4 neurons, respectively. The activation function is `torch.nn.CELU(alpha=1)`. The neural network controller is initialized as follows.

.. code-block:: python

    dual_controller = SingleFullyConnectedNeuralController(
        hidden_layers=[128, 64, 32, 16, 8, 4], activation=torch.nn.CELU(alpha=1)
    )

Training
--------

Although the neural network controller has not been trained yet, we can still utilize it to calculate the total cost if we apply this controller for 100 periods alongside our previously specified sourcing model.

.. code-block:: python

    dual_controller.get_total_cost(sourcing_model=dual_sourcing_model, sourcing_periods=100)

Unsurprisingly, the performance is poor because we are only using the untrained neural network in which the weights are just (pseudo) random numbers. We can train the neural network controller using the `train` method, in which the training data is generated from the given sourcing model. To better monitor the training process, we specify the `tensorboard_writer` parameter to log both the training loss and validation loss. For reproducibility, we also specify the seed of the underlying random number generator using the  `seed` parameter.

.. code-block:: python

    from torch.utils.tensorboard import SummaryWriter

    dual_controller.train(
        sourcing_model=dual_sourcing_model,
        sourcing_periods=100,
        validation_sourcing_periods=1000,
        epochs=2000,
        tensorboard_writer=SummaryWriter("runs/dual_sourcing_model"),
        seed=4,
    )

After training, we can use the trained neural network controller to calculate the total cost for 100 periods with our previously specified sourcing model. The total cost should be significantly lower than the cost associated with the untrained model.

.. code-block:: python
    
    dual_controller.get_total_cost(sourcing_model=dual_sourcing_model, sourcing_periods=100)

Simulation, Plotting and Order Calculation
------------------------------------------

We can also inspect how the controller performs in the specified sourcing environment by (i) plotting the inventory and order histories and (ii) calculating optimal orders.

.. code-block:: python

    # Simulate and plot the results
    dual_controller.plot(sourcing_model=dual_sourcing_model, sourcing_periods=100)
    # Calculate the optimal order quantity for applications
    regular_q, expedited_q = dual_controller.forward(
        current_inventory=torch.tensor([[10]]),
        past_regular_orders=torch.tensor([[1, 5]]),
        past_expedited_orders=torch.tensor([[0, 0]]),
    )

Save and Load the Model
-----------------------

It is also a good idea to save the trained neural network controller for future use. This can be done using the `save` method. The `load` method allows one to load a previously saved model.

.. code-block:: python

    # Save the model
    dual_controller.save("optimal_dual_sourcing_controller.pt")
    # Load the model
    dual_controller_loaded = DualFullyConnectedNeuralController().load("optimal_dual_sourcing_controller.pt")
---
title: 'idinn: A Python Package for Inventory-Dynamics Control with Neural Networks'
tags:
  - Python
  - PyTorch
  - artificial neural networks
  - inventory dynamics
  - optimization
  - control
  - dynamic programming
authors:
  - name: Jiawei Li
    affiliation: 1
    corresponding: true
  - name: Thomas Asikis
    orcid: 0000-0003-0163-4622
    affiliation: 2
  - name: Ioannis Fragkos
    affiliation: 3
  - name: Lucas B\"ottcher
    affiliation: "1,4"
    orcid: 0000-0003-1700-1897
affiliations:
 - name: Department of Computational Science and Philosophy, Frankfurt School of Finance and Management
   index: 1
 - name: Game Theory, University of Zurich
   index: 2
 - name: Department of Technology and Operations Management, Rotterdam School of Management, Erasmus University Rotterdam
   index: 3
 - name: Laboratory for Systems Medicine, Department of Medicine, University of Florida
   index: 4
date: 20 April 2024
bibliography: paper.bib

---

# Summary

Identifying optimal policies for replenishing inventory from multiple suppliers is a key problem in inventory management. Solving such optimization problems means that one must determine the quantities to order from each supplier based on the current net inventory and outstanding orders, minimizing the expected backlogging, holding, and sourcing costs. Despite over 60 years of extensive research on inventory management problems, even fundamental dual-sourcing problems [@barankin1961delivery,@fukuda1964optimal]—where orders from an expensive supplier arrive faster than orders from a regular supplier—remain analytically intractable. Additionally, there is a growing interest in optimization algorithms that are capable of handling real-world inventory problems with large numbers of suppliers and non-stationary demand.

We provide a Python package, `idinn`, implementing inventory dynamics–informed neural networks designed for controlling both single-sourcing and dual-sourcing problems. Neural network controllers and inventory dynamics are implemented into customizable objects with PyTorch backend, to enable users to find the optimal neural controllers for the user-specified inventory systems.

# Statement of need

Inventory management problems arise in almost all industries. A basic and yet analytically intractable problem in inventory management is dual sourcing [@barankin1961delivery,@fukuda1964optimal]. `idinn` is a Python package for controlling dual-sourcing inventory dynamics with dynamics-informed neural networks. Unlike traditional reinforcement-learning approaches, our optimization approach takes into account how the system being optimized behaves over time, leading to more efficient training and accurate solutions. 

Training neural networks for inventory-dynamics control presents a specific challenge. The adjustment of neural network weights during training relies on propagating real-valued gradients, whereas the neural network outputs—representing replenishment orders—must be integers. To address this challenge in optimizing a discrete problem with real-valued gradient descent learning algorithms, we employ a problem-tailored straight-through estimator [@yang2022injecting,@asikis2023multi]. This approach enables us to obtain integer-valued neural network outputs while backpropagating real-valued gradients.

`idinn` has been developed for researchers and students working at the intersection of optimization, operations research, and machine learning. It has been made available to students in a machine learning course at Frankfurt School to demonstrate the effectiveness of artificial neural networks in solving real-world optimization problems. In a previous publication [@bottcher2023control], a less accessible code base has been used to compute near-optimal solutions of dozens of dual-sourcing instances. 

# Example usage

## Solve single-sourcing problems using neural networks

Single-sourcing problems are inventory management problems where only one delivery option exists. The overall objective in single-sourcing and related inventory management problems is for companies to identify the optimal order quantities to minimize costs given stochastic demand. This problem can be addressed using `idinn`. We first initialize the sourcing model and its associated neural network controller. Subsequently, we train the neural network controller using data generated from the sourcing model. Finally, we can use the trained neural network controller to compute optimal order quantities.

### Initialization

Since we deal with the single-sourcing problem, we use the `SingleSourcingModel` class to initialize the sourcing model. Let us pick a single sourcing model which has a lead time of 0 (i.e., the order arrives immediately after it is placed), an initial inventory of 10 and a batch size of 32. The holding cost, $h$, and the shortage cost, $s$, are 5 and 495, respectively. The demand is generated from a uniform distribution with support $[1, 4]$. Notice that both the `demand_low` and `demand_low` parameters are inclusive. Hence, the generated demand will never exceed 4. In `idinn`, the sourcing model is initialized as follows.

```python
  import torch
  from idinn.sourcing_model import SingleSourcingModel
  from idinn.controller import SingleSourcingNeuralController

  single_sourcing_model = SingleSourcingModel(
    lead_time=0,
    holding_cost=5,
    shortage_cost=495,
    batch_size=32,
    init_inventory=10,
    demand_distribuion="uniform",
    demand_low=1,
    demand_high=4
  )
```

The cost at period $t$, $c_t$, is

$$
c_t = h \max(0, I_t) + s \max(0, - I_t)\,,
$$

where $I_t$ is the inventory level at period $t$. The higher the holding cost, the more costly it is to keep the inventory (when the inventory level is positive). The higher the shortage cost, the more costly it is to run out of stock (when the inventory level is negative). The cost can be calculated using the `get_cost` method of the sourcing model.

```python    
  single_sourcing_model.get_cost()
```

In our example, this function should return 50 for each sample since the initial inventory is 10 and the holding cost is 5. We have 32 samples in this case, as we specified a batch size of 32.

For single-sourcing problems, we initialize the neural network controller using the `SingleSourcingNeuralController` class. For illustration, we use a simple neural network with 1 hidden layer and 2 neurons. The activation function is `torch.nn.CELU(alpha=1)`. The neural network controller is initialized as follows.

```python
single_controller = SingleSourcingNeuralController(
    hidden_layers=[2], activation=torch.nn.CELU(alpha=1)
)
```

### Training

Although the neural network controller has not been trained yet, we can still compute the total cost associated with its order policy. To do so, we integrate it with our previously specified sourcing model and run simulations for 100 periods.

```python    
single_controller.get_total_cost(sourcing_model=single_sourcing_model, sourcing_periods=100)
```

Unsurprisingly, the performance is poor because we are only using the untrained neural network in which the weights are just (pseudo) random numbers. We can train the neural network controller using the `train` method, in which the training data is generated from the given sourcing model. To better monitor the training process, we specify the `tensorboard_writer` parameter to log both the training loss and validation loss. For reproducibility, we also specify the seed of the underlying random number generator using the `seed` parameter.

```python
from torch.utils.tensorboard import SummaryWriter

single_controller.train(
    sourcing_model=sourcing_model,
    sourcing_periods=50,
    validation_sourcing_periods=1000,
    epochs=5000,
    seed=1,
    tensorboard_writer=SummaryWriter()
)
```

After training, we can use the trained neural network controller to calculate the total cost for 100 periods with our previously specified sourcing model. The total cost should be significantly lower than the cost associated with the untrained model.

```python
single_controller.get_total_cost(sourcing_model=single_sourcing_model, sourcing_periods=100)
```

### Simulation, plotting, and order calculation

We can also inspect how the controller performs in the specified sourcing environment by plotting the inventory and order histories and calculating optimal orders.

```python
# Simulate and plot the results
single_controller.plot(sourcing_model=single_sourcing_model, sourcing_periods=100)
# Calculate the optimal order quantity for applications
single_controller.forward(current_inventory=10, past_orders=[1, 5])
```

### Save and load the model

It is also a good idea to save the trained neural network controller for future use. This can be done using the `save` method. The `load` method allows the user to load a previously saved controller.

```python
# Save the model
single_controller.save("optimal_single_sourcing_controller.pt")
# Load the model
single_controller_loaded = SingleSourcingNeuralController(
    hidden_layers=[2], activation=torch.nn.CELU(alpha=1)
)
single_controller_loaded.load("optimal_single_sourcing_controller.pt")
```

## Solve dual-sourcing problems using neural networks

Dual-sourcing problems are similar to single-sourcing problems but are more intricate. In a dual-sourcing problem, a company has two potential suppliers for a product, each offering varying lead times (the duration for orders to arrive) and order costs (the expense of placing an order). The challenge lies in the company's decision-making process: determining which supplier to engage for each product to minimize costs given stochastic demand. We can solve dual-sourcing problems with `idinn` in a way similar to the solution to single-sourcing problems described in the previous section.

### Initialization

To address dual-sourcing problems, we employ two main classes: `DualSourcingModel` and `DualSourcingNeuralController`, responsible for setting up the sourcing model and its corresponding controller. In this example, we examine a dual-sourcing model characterized by the following parameters: both regular order lead time and expedited order lead time are set to 0; the regular order cost, $c^r$, is 0; the expedited order cost, $c^e$, is 20; the initial inventory is 6, and the batch size is 256. Additionally, the holding cost, $h$, and the shortage cost, $s$, are 5 and 495, respectively. Demand is generated from a uniform distribution with interval $[1, 4]$. Notice that both the `demand_low` and `demand_low` parameters are inclusive. Hence, the generated demand will never exceed 4. In `idinn`, the sourcing model is initialized as follows.

```python    
import torch
from idinn.sourcing_model import DualSourcingModel
from idinn.controller import DualSourcingNeuralController

dual_sourcing_model = DualSourcingModel(
    regular_lead_time=2,
    expedited_lead_time=0,
    regular_order_cost=0,
    expedited_order_cost=20,
    holding_cost=5,
    shortage_cost=495,
    batch_size=256,
    init_inventory=6,
    demand_distribuion="uniform",
    demand_low=1,
    demand_high=4
)
```

The cost at period `t`, `c_t`, is

$$
c_t = c^r q^r_t + c^e q^e_t + h \max(0, I_t) + s \max(0, - I_t)\,,
$$

where $I_t$ is the inventory level at period $t$, $q^r_t$ is the regular order placed at period $t$, and $q^e_t$ is the expedited order placed at period $t$. The higher the holding cost, the more costly it is to keep the inventory (when the inventory level is positive). The higher the shortage cost, the more costly it is to run out of stock (when the inventory level is negative). The higher the regular or expedited order costs, the more costly it is to place the respective orders. The cost can be calculated using the `get_cost` method of the sourcing model.

```python    
dual_sourcing_model.get_cost(regular_q=0, expedited_q=0)
```

In our example, this function should return 30 for each sample since the initial inventory is 6, the holding cost is 5, and there is neither a regular nor expedited order. We have 256 samples in this case, as we specified a batch size of 256.

For dual-sourcing problems, we initialize the neural network controller using the `DualSourcingNeuralController` class. We use a simple neural network with 6 hidden layers. The numbers of neurons in each layer are 128, 64, 32, 16, 8, and 4, respectively. The activation function is `torch.nn.CELU(alpha=1)`. The neural network controller is initialized as follows.

```python
dual_controller = DualSourcingNeuralController(
    hidden_layers=[128, 64, 32, 16, 8, 4], activation=torch.nn.CELU(alpha=1)
)
```

### Training

Although the neural network controller has not been trained yet, we can still utilize it to calculate the total cost if we apply this controller for 100 periods alongside our previously specified sourcing model.

```python
dual_controller.get_total_cost(sourcing_model=dual_sourcing_model, sourcing_periods=100)
```

Unsurprisingly, the performance is poor because we are only using the untrained neural network in which the weights are just (pseudo) random numbers. We can train the neural network controller using the `train` method, in which the training data is generated from the given sourcing model. To better monitor the training process, we specify the `tensorboard_writer` parameter to log both the training loss and validation loss. For reproducibility, we also specify the seed of the underlying random number generator using the `seed` parameter.

```python
from torch.utils.tensorboard import SummaryWriter

dual_controller.train(
    sourcing_model=dual_sourcing_model,
    sourcing_periods=100,
    validation_sourcing_periods=1000,
    epochs=2000,
    tensorboard_writer=SummaryWriter("runs/dual_sourcing_model"),
    seed=4
)
```

After training, we can use the trained neural network controller to calculate the total cost for 100 periods with our previously specified sourcing model. The total cost should be significantly lower than the cost associated with the untrained model.

```python    
dual_controller.get_total_cost(sourcing_model=dual_sourcing_model, sourcing_periods=100)
```

### Simulation, plotting, and order calculation

We can also inspect how the controller performs in the specified sourcing environment by plotting the inventory and order histories, and calculating optimal orders.

```python
# Simulate and plot the results
dual_controller.plot(sourcing_model=dual_sourcing_model, sourcing_periods=100)
# Calculate the optimal order quantity for applications
regular_q, expedited_q = dual_controller.forward(
    current_inventory=10,
    past_regular_orders=[1, 5],
    past_expedited_orders=[0, 0],
)
```

### Save and load the model

It is also a good idea to save the trained neural network controller for future use. This can be done using the `save` method. The `load` method allows one to load a previously saved model.

```python
# Save the model
dual_controller.save("optimal_dual_sourcing_controller.pt")
# Load the model
dual_controller_loaded = DualSourcingNeuralController(
    hidden_layers=[128, 64, 32, 16, 8, 4], activation=torch.nn.CELU(alpha=1)
)
dual_controller_loaded.load("optimal_dual_sourcing_controller.pt")
```

# Acknowledgements

LB acknowledges financial support from hessian.AI and the Army Research Office (grant W911NF-23-1-0129). TA acknowledges financial support from the Schweizerischer Nationalfonds zur Förderung der Wissenschaf­tlichen Forschung (NCCR Automation) (grant P2EZP2 191888).


# References
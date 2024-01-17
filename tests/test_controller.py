import torch
from torch import tensor
import matplotlib.pyplot as plt
from idinn.sourcing_model import SingleSourcingModel, DualSourcingModel
from idinn.controller import SingleFullyConnectedNeuralController, DualFullyConnectedNeuralController, CappedDualIndexController

def test_single_controller():
    # Create an instance of the controller
    hidden_layers = [4, 2]
    activation = torch.nn.ReLU()
    controller = SingleFullyConnectedNeuralController(hidden_layers, activation)

    # Test init_layers method
    lead_time = 2
    controller.init_layers(lead_time)
    assert len(controller.stack) == len(hidden_layers) * 2 + 2

    # Test forward method
    current_inventory = torch.tensor([[10.]])
    past_orders = [tensor([[0.]]), tensor([[0.]])]
    q = controller.forward(current_inventory, past_orders)
    assert q.shape == torch.Size([1, 1])

    # Test get_total_cost method
    single_sourcing_model = SingleSourcingModel(
        lead_time=2,
        holding_cost=0.5,
        shortage_cost=1,
        init_inventory=10,
        batch_size=1,
        demand_generator=torch.distributions.Uniform(low=0, high=5)
    )
    sourcing_periods = 5
    total_cost = controller.get_total_cost(single_sourcing_model, sourcing_periods, seed=42)
    assert total_cost.item() == 11.5

    # Test simulate method
    past_inventories, past_orders = controller.simulate(single_sourcing_model, sourcing_periods, seed=42)
    assert past_inventories[0] == 10
    assert past_orders[0] == 0

    # Test train method
    sourcing_periods = 5
    controller.train(single_sourcing_model, sourcing_periods, epochs=1)

    # Test plot method
    sourcing_periods = 5
    controller.plot(single_sourcing_model, sourcing_periods)
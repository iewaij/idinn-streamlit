import pytest
import torch
from torch import tensor
import matplotlib.pyplot as plt
from idinn.sourcing_model import SingleSourcingModel, DualSourcingModel
from idinn.controller import SingleSourcingNeuralController, DualSourcingNeuralController

@pytest.fixture
def single_sourcing_model():
    return SingleSourcingModel(
        lead_time=2,
        holding_cost=0.5,
        shortage_cost=1,
        init_inventory=10,
        batch_size=1,
        demand_generator=torch.distributions.Uniform(low=0, high=5)
    )

@pytest.fixture
def single_sourcing_controller():
    single_sourcing_controller = SingleSourcingNeuralController(
        hidden_layers = [4, 2],
        activation = torch.nn.ReLU()
    )
    single_sourcing_controller.init_layers(lead_time=2)
    return single_sourcing_controller

def test_single_controller_init_layers(single_sourcing_controller: SingleSourcingNeuralController):
    assert len(single_sourcing_controller.stack) == 6

def test_single_controller_forward(single_sourcing_controller: SingleSourcingNeuralController):
    current_inventory = torch.tensor([[10.]])
    past_orders = tensor([[0., 0.]])
    q = single_sourcing_controller.forward(current_inventory, past_orders)
    assert q.shape == torch.Size([1, 1])

def test_single_controller_get_total_cost(single_sourcing_model: SingleSourcingModel, single_sourcing_controller: SingleSourcingNeuralController):
    total_cost = single_sourcing_controller.get_total_cost(single_sourcing_model, sourcing_periods=5, seed=42)
    assert total_cost.item() == 11.5

def test_single_controller_simulate(single_sourcing_model: SingleSourcingModel, single_sourcing_controller: SingleSourcingNeuralController):
    past_inventories, past_orders = single_sourcing_controller.simulate(single_sourcing_model, sourcing_periods=5, seed=42)
    assert past_inventories[0] == 10
    assert past_orders[0] == 0

def test_single_controller_train(single_sourcing_model: SingleSourcingModel, single_sourcing_controller: SingleSourcingNeuralController):
    single_sourcing_controller.train(single_sourcing_model, sourcing_periods=5, epochs=1)

def test_single_controller_plot(single_sourcing_model: SingleSourcingModel, single_sourcing_controller: SingleSourcingNeuralController):
    single_sourcing_controller.plot(single_sourcing_model, sourcing_periods=5)

@pytest.fixture
def dual_sourcing_model():
    """
    Dual sourcing model at default state.
    """
    return DualSourcingModel(
        regular_lead_time=2,
        expedited_lead_time=1,
        regular_order_cost=0,
        expedited_order_cost=20,
        holding_cost=0.5,
        shortage_cost=1,
        init_inventory=10,
        batch_size=1,
    )

@pytest.fixture
def dual_sourcing_controller():
    dual_sourcing_controller = DualSourcingNeuralController(
        hidden_layers = [4, 2],
        activation = torch.nn.ReLU()
    )
    dual_sourcing_controller.init_layers(regular_lead_time=2, expedited_lead_time=1)
    return dual_sourcing_controller
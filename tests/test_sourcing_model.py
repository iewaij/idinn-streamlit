import torch
import pytest
from torch import tensor
from idinn.sourcing_model import SingleSourcingModel, DualSourcingModel

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
def dual_sourcing_model():
    return DualSourcingModel(
        regular_lead_time=2,
        expedited_lead_time=1,
        regular_order_cost=10,
        expedited_order_cost=20,
        holding_cost=0.5,
        shortage_cost=1,
        init_inventory=10,
        batch_size=1,
        demand_generator=torch.distributions.Uniform(low=0, high=5)
    )

def test_single_sourcing_model(single_sourcing_model):
    assert single_sourcing_model.get_lead_time() == 2
    assert single_sourcing_model.get_past_orders() == [tensor([[0.]]), tensor([[0.]])]
    assert single_sourcing_model.get_cost() == tensor([[5.]])

    single_sourcing_model.order(tensor([[5.]]), seed=42)
    assert single_sourcing_model.get_past_orders() == [tensor([[0.]]), tensor([[0.]]), tensor([[5.]])]
    assert single_sourcing_model.get_cost() == tensor([[3.]])

def test_dual_sourcing_model(dual_sourcing_model):
    assert dual_sourcing_model.get_regular_lead_time() == 2
    assert dual_sourcing_model.get_expedited_lead_time() == 1
    assert dual_sourcing_model.get_past_regular_orders() == [tensor([[0.]]), tensor([[0.]])]
    assert dual_sourcing_model.get_past_expedited_orders() == [tensor([[0.]])]
    assert dual_sourcing_model.get_cost(5, 5) == tensor([[155.]])

    dual_sourcing_model.order(tensor([[3.]]), tensor([[2.]]), seed=42)
    assert dual_sourcing_model.get_past_regular_orders() == [tensor([[0.]]), tensor([[0.]]), tensor([[3.]])]
    assert dual_sourcing_model.get_past_expedited_orders() == [tensor([[0.]]), tensor([[2.]])]
    assert dual_sourcing_model.get_cost(5, 5) == tensor([[153.]])
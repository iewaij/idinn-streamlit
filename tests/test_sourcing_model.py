import torch
import pytest
from idinn.sourcing_model import SingleSourcingModel, DualSourcingModel


@pytest.fixture
def single_sourcing_model():
    """
    Single sourcing model at default state.
    """
    return SingleSourcingModel(
        lead_time=2,
        holding_cost=0.5,
        shortage_cost=1,
        init_inventory=10,
        batch_size=3,
    )


def test_single_sourcing_model(single_sourcing_model: SingleSourcingModel):
    assert single_sourcing_model.get_lead_time() == 2
    assert torch.all(torch.eq(single_sourcing_model.get_init_inventory(), torch.tensor([10.])))
    assert torch.all(torch.eq(single_sourcing_model.get_past_orders(), torch.zeros(3, 3)))
    assert torch.all(torch.eq(single_sourcing_model.get_past_inventories(), torch.tensor([10.]).repeat(3, 3)))
    assert torch.all(torch.eq(single_sourcing_model.get_current_inventory(), torch.tensor([[10.], [10.], [10.]])))
    assert torch.all(torch.eq(single_sourcing_model.get_cost(), torch.tensor([[5.], [5.], [5.]])))

@pytest.fixture
def single_sourcing_model_fixed_demand():
    """
    Single sourcing model with fixed demand at 1.
    """
    return SingleSourcingModel(
        lead_time=1,
        holding_cost=0.5,
        shortage_cost=1,
        init_inventory=10,
        batch_size=3,
        demand_generator=torch.distributions.Uniform(low=1, high=2),
    )

def test_single_sourcing_model_fixed_demand(single_sourcing_model_fixed_demand: SingleSourcingModel):
    single_sourcing_model_fixed_demand.order(torch.tensor([[1.], [2.], [3.]]))
    assert torch.all(torch.eq(single_sourcing_model_fixed_demand.get_past_orders(), torch.cat([torch.zeros(3, 2), torch.tensor([[1.], [2.], [3.]])], dim=1)))
    assert torch.all(torch.eq(single_sourcing_model_fixed_demand.get_past_inventories(), torch.cat([torch.tensor([10.]).repeat(3, 2), torch.tensor([[9.], [9.], [9.]])], dim=1)))
    assert torch.all(torch.eq(single_sourcing_model_fixed_demand.get_current_inventory(), torch.tensor([[9.], [9.], [9.]])))
    assert torch.all(torch.eq(single_sourcing_model_fixed_demand.get_cost(), torch.tensor([[4.5], [4.5], [4.5]])))

    single_sourcing_model_fixed_demand.order(torch.tensor([[1.], [2.], [3.]]))
    assert torch.all(torch.eq(single_sourcing_model_fixed_demand.get_past_orders(), torch.cat([torch.zeros(3, 2), torch.tensor([[1.,1.], [2.,2.], [3.,3.]])], dim=1)))
    assert torch.all(torch.eq(single_sourcing_model_fixed_demand.get_current_inventory(), torch.tensor([[9.], [10.], [11.]])))
    assert torch.all(torch.eq(single_sourcing_model_fixed_demand.get_cost(), torch.tensor([[4.5], [5.], [5.5]])))

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
        batch_size=3,
    )


def test_dual_sourcing_model(dual_sourcing_model):
    assert dual_sourcing_model.get_expedited_lead_time() == 1
    assert dual_sourcing_model.get_regular_lead_time() == 2
    assert torch.all(torch.eq(dual_sourcing_model.get_init_inventory(), torch.tensor([10.])))
    assert torch.all(torch.eq(dual_sourcing_model.get_past_regular_orders(), torch.zeros(3, 3)))
    assert torch.all(torch.eq(dual_sourcing_model.get_past_expedited_orders(), torch.zeros(3, 3)))
    assert torch.all(torch.eq(dual_sourcing_model.get_past_inventories(), torch.tensor([10.]).repeat(3, 3)))
    assert torch.all(torch.eq(dual_sourcing_model.get_current_inventory(), torch.tensor([[10.], [10.], [10.]])))
    assert torch.all(torch.eq(dual_sourcing_model.get_cost(0, 0), torch.tensor([[5.], [5.], [5.]])))


@pytest.fixture
def dual_sourcing_model_fixed_demand():
    """
    Dual sourcing model with fixed demand at 1.
    """
    return DualSourcingModel(
        regular_lead_time=1,
        expedited_lead_time=0,
        regular_order_cost=0,
        expedited_order_cost=20,
        holding_cost=0.5,
        shortage_cost=1,
        init_inventory=10,
        batch_size=3,
        demand_generator=torch.distributions.Uniform(low=1, high=2),
    )

def test_dual_sourcing_model_fixed_demand(dual_sourcing_model_fixed_demand: DualSourcingModel):
    dual_sourcing_model_fixed_demand.order(regular_q=torch.tensor([[1.], [2.], [3.]]), expedited_q=torch.tensor([[4.], [5.], [6.]]))
    assert torch.all(torch.eq(
        dual_sourcing_model_fixed_demand.get_past_regular_orders(),
        torch.cat([torch.zeros(3, 2), torch.tensor([[1.], [2.], [3.]])], dim=1)
        ))
    assert torch.all(torch.eq(
        dual_sourcing_model_fixed_demand.get_past_expedited_orders(),
        torch.cat([torch.zeros(3, 2), torch.tensor([[4.], [5.], [6.]])], dim=1)
        ))
    assert torch.all(torch.eq(
        dual_sourcing_model_fixed_demand.get_past_inventories(),
        torch.cat([torch.tensor([10.]).repeat(3, 2), torch.tensor([[13.], [14.], [15.]])], dim=1)
        ))
    assert torch.all(torch.eq(
        dual_sourcing_model_fixed_demand.get_current_inventory(),
        torch.tensor([[13.], [14.], [15.]])
        ))
    assert torch.all(torch.eq(
        dual_sourcing_model_fixed_demand.get_cost(regular_q=0, expedited_q=1),
        torch.tensor([[26.5], [27.], [27.5]])
        ))
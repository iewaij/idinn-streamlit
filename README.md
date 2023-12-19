# Inventory Optimization

```python
from torch.utils.tensorboard import SummaryWriter
from sourcing_model import SingleSourcingModel
from neural_controller import SingleFullyConnectedRegressionController

controller = SingleFullyConnectedRegressionController()

sourcing_model = SingleSourcingModel(
    lead_time=0,
    holding_cost=5,
    shortage_cost=495,
    batch_size=32,
    init_inventory=0
)

validation_sourcing_model = SingleSourcingModel(
    lead_time=0,
    holding_cost=5,
    shortage_cost=495,
    batch_size=32,
    init_inventory=0
)

writer = SummaryWriter()

controller.train(
    sourcing_model=sourcing_model,
    sourcing_periods=50,
    validation_sourcing_model=validation_sourcing_model,
    validation_sourcing_periods=100,
    epochs=5000,
    writer=writer
)
```
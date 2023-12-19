import torch
from torch import nn


class SingleFullyConnectedRegressionController(nn.Module):
    def __init__(self):
        super().__init__()

    def init_layers(self, lead_time):
        # TODO: Find a better way to init layers
        self.lead_time = lead_time
        self.stack = torch.nn.Sequential(
            torch.nn.Linear(lead_time + 1, 2),
            torch.nn.CELU(alpha=1),
            torch.nn.Linear(2, 1, bias=False),
            torch.nn.ReLU(),
        )

    def forward(
        self,
        current_inventory,
        past_orders,
    ):
        obs_list = [current_inventory]
        if self.lead_time > 0:
            if isinstance(past_orders, list):
                order_obs = torch.cat(past_orders[-self.lead_time :], dim=-1)
            else:
                order_obs = past_orders[:, -self.lead_time :]
            obs_list.append(order_obs)

        h = torch.cat(obs_list, dim=-1)
        h = self.stack(h)
        h = h - torch.frac(h).clone().detach()
        q = h[:, 0].unsqueeze(-1)
        return q

    def loss_fn(self, sourcing_model, sourcing_periods, writer=None):
        total_cost = 0
        for i in range(sourcing_periods):
            current_inventory = sourcing_model.get_inventory()
            past_orders = sourcing_model.get_past_orders()
            q = self.forward(current_inventory, past_orders)

            if writer is not None:
                writer.add_scalar("Sourcing/inventory", current_inventory[-1], i)
                writer.add_scalar("Sourcing/order", q[-1], i)

            sourcing_model.order(q)
            current_costs = sourcing_model.get_cost()
            total_cost += current_costs.mean()

        return total_cost

    def train(
        self,
        sourcing_model,
        sourcing_periods,
        validation_sourcing_model,
        validation_sourcing_periods,
        epochs,
        writer=None,
    ):
        lead_time = sourcing_model.get_lead_time()
        self.init_layers(lead_time)

        optimizer_init_inventory = torch.optim.RMSprop(
            [sourcing_model.init_inventory], lr=1e-1
        )
        optimizer_parameters = torch.optim.RMSprop(self.parameters(), lr=3e-3)

        for epoch in range(epochs):
            # Reset the sourcing model with the learned init inventory
            sourcing_model.reset()

            optimizer_parameters.zero_grad()
            optimizer_init_inventory.zero_grad()

            # Log the order and inventory
            if writer is not None and epoch == epochs - 1:
                total_cost = self.loss_fn(sourcing_model, sourcing_periods, writer)

            total_cost = self.loss_fn(sourcing_model, sourcing_periods)
            total_cost.backward()

            # Log train loss
            if writer is not None:
                writer.add_scalar("Loss/train", total_cost, epoch)

            # Gradient descend
            optimizer_parameters.step()
            if epoch % 3 == 0:
                optimizer_init_inventory.step()

            # Log evaluation loss
            if epoch % 10 == 0:
                eval_cost = self.loss_fn(
                    validation_sourcing_model, validation_sourcing_periods
                )
                writer.add_scalar("Loss/eval", eval_cost, epoch)

        writer.flush()

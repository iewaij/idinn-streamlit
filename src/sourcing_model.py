import torch


class SingleSourcingModel(torch.nn.Module):
    def __init__(
        self,
        lead_time,
        holding_cost,
        shortage_cost,
        init_inventory,
        batch_size,
        demand_generator=torch.distributions.Uniform(low=0, high=5),
    ):
        super().__init__()
        self.lead_time = lead_time
        self.holding_cost = holding_cost
        self.shortage_cost = shortage_cost
        self.batch_size = batch_size
        self.init_inventory = torch.tensor(
            [init_inventory], requires_grad=True, dtype=torch.float
        )
        self.inventory = (
            self.init_inventory.repeat([self.batch_size, 1])
            - torch.frac(self.init_inventory).clone().detach()
        )
        self.demand_generator = demand_generator

        if self.lead_time == 0:
            self.past_orders = [torch.zeros([self.batch_size, 1])]
        elif self.lead_time > 0:
            self.past_orders = [torch.zeros([self.batch_size, 1])] * self.lead_time

    def get_lead_time(self):
        return self.lead_time

    def get_inventory(self):
        return self.inventory

    def get_past_orders(self):
        return self.past_orders

    def get_cost(self):
        current_costs = self.holding_cost * torch.relu(
            self.inventory
        ) + self.shortage_cost * torch.relu(-self.inventory)
        return current_costs

    def order(self, q):
        # Current orders are added to past_orders
        self.past_orders.append(q)
        # Past orders arrived
        arrived_orders = self.past_orders[-self.lead_time]
        # Generate demands in integer
        current_demands = self.demand_generator.sample([self.batch_size, 1]).int()
        # Update inventory
        self.inventory = self.inventory + arrived_orders - current_demands

    def reset(self):
        self.inventory = (
            self.init_inventory.repeat([self.batch_size, 1])
            - torch.frac(self.init_inventory).clone().detach()
        )

        if self.lead_time == 0:
            self.past_orders = [torch.zeros([self.batch_size, 1])]
        elif self.lead_time > 0:
            self.past_orders = [torch.zeros([self.batch_size, 1])] * self.lead_time

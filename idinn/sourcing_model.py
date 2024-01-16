import torch


class BaseSourcingModel:
    def __init__(
        self,
        holding_cost,
        shortage_cost,
        init_inventory,
        batch_size,
        demand_generator=torch.distributions.Uniform(low=0, high=5),
    ):
        self.holding_cost = holding_cost
        self.shortage_cost = shortage_cost
        self.batch_size = batch_size
        self.init_inventory = torch.tensor(
            [init_inventory], requires_grad=True, dtype=torch.float
        )
        self.past_inventories = [self.init_inventory]
        self.demand_generator = demand_generator

    def get_init_inventory(self):
        init_inventory = (
            self.init_inventory.repeat([self.batch_size, 1])
            - torch.frac(self.init_inventory).clone().detach()
        )
        return init_inventory

    def get_past_inventories(self):
        return self.past_inventories

    def get_current_inventory(self):
        return self.past_inventories[-1]


class SingleSourcingModel(BaseSourcingModel):
    def __init__(
        self,
        lead_time,
        holding_cost,
        shortage_cost,
        init_inventory,
        batch_size=1,
        demand_generator=torch.distributions.Uniform(low=0, high=5),
    ):
        """
        Parameters
        ----------
        lead_time : float
            The lead time for orders.
        holding_cost : float
            The cost of holding inventory.
        shortage_cost : float
            The cost of inventory shortage.
        init_inventory : float
            The initial inventory.
        batch_size : int, optional
            The batch size for orders. Default is 1.
        demand_generator : torch.distributions.Distribution, optional
            The demand generator for generating demand values. Default is a uniform distribution
            with low=0 and high=5, which is equivalent to torch.randint(0, 4).
        """
        super().__init__(
            holding_cost, shortage_cost, init_inventory, batch_size, demand_generator
        )
        self.lead_time = lead_time
        self.reset()

    def reset(self, batch_size=None):
        if batch_size is not None and self.batch_size != batch_size:
            self.batch_size = batch_size
        self.past_inventories = [self.get_init_inventory()]
        self.past_orders = [torch.zeros([self.batch_size, 1])]
        if self.lead_time > 0:
            self.past_orders = self.past_orders * self.lead_time

    def get_lead_time(self):
        return self.lead_time

    def get_past_orders(self):
        return self.past_orders

    def get_cost(self):
        cost = self.holding_cost * torch.relu(
            self.past_inventories[-1]
        ) + self.shortage_cost * torch.relu(-self.past_inventories[-1])
        return cost

    def order(self, q, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        # Current orders are added to past_orders
        self.past_orders.append(q)
        # Past orders arrived
        arrived_order = self.past_orders[-1 - self.lead_time]
        # Generate demands and floored to integer
        current_demand = self.demand_generator.sample([self.batch_size, 1]).int()
        # Update inventory
        current_inventory = self.past_inventories[-1] + arrived_order - current_demand
        self.past_inventories.append(current_inventory)


class DualSourcingModel(BaseSourcingModel):
    def __init__(
        self,
        regular_lead_time,
        expedited_lead_time,
        regular_order_cost,
        expedited_order_cost,
        holding_cost,
        shortage_cost,
        init_inventory,
        batch_size=1,
        demand_generator=torch.distributions.Uniform(low=0, high=5),
    ):
        """
        Parameters
        ----------
        regular_lead_time : float
            The lead time for regular orders.
        expedited_lead_time : float
            The lead time for expedited orders.
        regular_order_cost : float
            The cost of placing a regular order.
        expedited_order_cost : float
            The cost of placing an expedited order.
        holding_cost : float
            The cost of holding inventory.
        shortage_cost : float
            The cost of shortage.
        init_inventory : float
            The initial inventory.
        batch_size : int, optional
            The batch size for orders. Default is 1.
        demand_generator : torch.distributions.Distribution, optional
            The demand generator for generating demand values. Default is a uniform distribution
            with low=0 and high=5 which is equivalent to torch.randint(0, 4).
        """
        super().__init__(
            holding_cost, shortage_cost, init_inventory, batch_size, demand_generator
        )
        self.regular_lead_time = regular_lead_time
        self.expedited_lead_time = expedited_lead_time
        self.regular_order_cost = regular_order_cost
        self.expedited_order_cost = expedited_order_cost
        self.reset()

    def reset(self, batch_size=None):
        if batch_size is not None and self.batch_size != batch_size:
            self.batch_size = batch_size
        self.past_inventories = [self.get_init_inventory()]
        self.past_regular_orders = [torch.zeros([self.batch_size, 1])]
        self.past_expedited_orders = [torch.zeros([self.batch_size, 1])]

        if self.regular_lead_time > 0:
            self.past_regular_orders = self.past_regular_orders * self.regular_lead_time
        if self.expedited_lead_time > 0:
            self.past_expedited_orders = (
                self.past_expedited_orders * self.expedited_lead_time
            )

    def get_past_regular_orders(self):
        return self.past_regular_orders

    def get_past_expedited_orders(self):
        return self.past_expedited_orders

    def get_regular_lead_time(self):
        return self.regular_lead_time

    def get_expedited_lead_time(self):
        return self.expedited_lead_time

    def get_cost(self, regular_q, expedited_q):
        cost = (
            self.regular_order_cost * regular_q
            + self.expedited_order_cost * expedited_q
            + self.holding_cost * torch.relu(self.past_inventories[-1])
            + self.shortage_cost * torch.relu(-self.past_inventories[-1])
        )
        return cost

    def order(self, regular_q, expedited_q, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        # Current regular order are added to past_regular_orders
        self.past_regular_orders.append(regular_q)
        # Current expedited order are added to past_expedited_orders
        self.past_expedited_orders.append(expedited_q)
        # Past expedited orders arrived
        arrived_expedited_orders = self.past_expedited_orders[
            -1 - self.expedited_lead_time
        ]
        # Past orders arrived
        arrived_regular_orders = self.past_regular_orders[-1 - self.regular_lead_time]
        # Generate demands and floored to integer
        current_demand = self.demand_generator.sample([self.batch_size, 1]).int()
        # Update inventory
        current_inventory = (
            self.past_inventories[-1]
            + arrived_expedited_orders
            + arrived_regular_orders
            - current_demand
        )
        self.past_inventories.append(current_inventory)

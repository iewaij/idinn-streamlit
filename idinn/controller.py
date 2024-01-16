import matplotlib.pyplot as plt
import numpy as np
import torch


class NeuralControllerMixIn:
    def save(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)

    def resume(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path))


class SingleFullyConnectedNeuralController(torch.nn.Module, NeuralControllerMixIn):
    def __init__(self, hidden_layers=[2], activation=torch.nn.CELU(alpha=1)):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.activation = activation

    def init_layers(self, lead_time):
        self.lead_time = lead_time
        architecture = [
            torch.nn.Linear(lead_time + 1, self.hidden_layers[0]),
            self.activation,
        ]
        for i in range(len(self.hidden_layers)):
            if i < len(self.hidden_layers) - 1:
                architecture += [
                    torch.nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]),
                    self.activation,
                ]
        architecture += [
            torch.nn.Linear(self.hidden_layers[-1], 1, bias=False),
            torch.nn.ReLU(),
        ]
        self.stack = torch.nn.Sequential(*architecture)

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

    def get_total_cost(self, sourcing_model, sourcing_periods, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        total_cost = 0
        for i in range(sourcing_periods):
            current_inventory = sourcing_model.get_current_inventory()
            past_orders = sourcing_model.get_past_orders()
            q = self.forward(current_inventory, past_orders)
            sourcing_model.order(q)
            current_cost = sourcing_model.get_cost()
            total_cost += current_cost.mean()
        return total_cost

    def train(
        self,
        sourcing_model,
        sourcing_periods,
        epochs,
        validation_sourcing_periods=None,
        lr_init_inventory=1e-1,
        lr_parameters=3e-3,
        seed=None,
        tensorboard_writer=None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
        lead_time = sourcing_model.get_lead_time()
        self.init_layers(lead_time)

        optimizer_init_inventory = torch.optim.RMSprop(
            [sourcing_model.init_inventory], lr=lr_init_inventory
        )
        optimizer_parameters = torch.optim.RMSprop(self.parameters(), lr=lr_parameters)
        min_cost = np.inf

        for epoch in range(epochs):
            # Clear grad cache
            optimizer_parameters.zero_grad()
            optimizer_init_inventory.zero_grad()
            # Reset the sourcing model with the learned init inventory
            sourcing_model.reset()
            total_cost = self.get_total_cost(sourcing_model, sourcing_periods)
            total_cost.backward()
            # Gradient descend
            if epoch % 3 == 0:
                optimizer_init_inventory.step()
            else:
                optimizer_parameters.step()
            # Save the best model
            if total_cost < min_cost:
                best_state = self.state_dict()
            # Log train loss
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar(
                    "Avg. cost per period/train", total_cost / sourcing_periods, epoch
                )
                # Log evaluation loss
                if validation_sourcing_periods is not None and epoch % 10 == 0:
                    eval_cost = self.get_total_cost(
                        sourcing_model, validation_sourcing_periods
                    )
                    tensorboard_writer.add_scalar(
                        "Avg. cost per period/eval",
                        eval_cost / validation_sourcing_periods,
                        epoch,
                    )
                tensorboard_writer.flush()

        self.load_state_dict(best_state)

    def simulate(self, sourcing_model, sourcing_periods, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        sourcing_model.reset(batch_size=1)
        for i in range(sourcing_periods):
            current_inventory = sourcing_model.get_current_inventory()
            past_orders = sourcing_model.get_past_orders()
            q = self.forward(current_inventory, past_orders)
            sourcing_model.order(q)
        past_inventories = sourcing_model.get_past_inventories()
        past_inventories = list(map(int, past_inventories))
        past_orders = sourcing_model.get_past_orders()
        past_orders = list(map(int, past_orders))
        return past_inventories, past_orders

    def plot(self, sourcing_model, sourcing_periods):
        past_inventories, past_orders = self.simulate(
            sourcing_model=sourcing_model, sourcing_periods=sourcing_periods
        )
        fig, ax = plt.subplots(ncols=2, figsize=(10, 4))

        ax[0].step(range(sourcing_periods), past_inventories[-sourcing_periods:])
        ax[0].yaxis.get_major_locator().set_params(integer=True)
        ax[0].set_title("Inventory")
        ax[0].set_xlabel("Period")
        ax[0].set_ylabel("Quantity")

        ax[1].step(range(sourcing_periods), past_orders[-sourcing_periods:])
        ax[1].yaxis.get_major_locator().set_params(integer=True)
        ax[1].set_title("Order")
        ax[1].set_xlabel("Period")
        ax[1].set_ylabel("Quantity")


class DualFullyConnectedNeuralController(torch.nn.Module, NeuralControllerMixIn):
    def __init__(
        self,
        hidden_layers=[128, 64, 32, 16, 8, 4],
        activation=torch.nn.CELU(alpha=1),
        compressed=False,
    ):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.compressed = compressed

    def init_layers(self, regular_lead_time, expedited_lead_time):
        self.regular_lead_time = regular_lead_time
        self.expedited_lead_time = expedited_lead_time
        if self.compressed:
            input_length = regular_lead_time + expedited_lead_time
        else:
            input_length = regular_lead_time + expedited_lead_time + 1

        architecture = [
            torch.nn.Linear(input_length, self.hidden_layers[0]),
            self.activation,
        ]
        for i in range(len(self.hidden_layers)):
            if i < len(self.hidden_layers) - 1:
                architecture += [
                    torch.nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]),
                    self.activation,
                ]
        architecture += [
            torch.nn.Linear(self.hidden_layers[-1], 2),
            torch.nn.ReLU(),
        ]
        self.stack = torch.nn.Sequential(*architecture)

    def forward(self, current_inventory, past_regular_orders, past_expedited_orders):
        if self.compressed:
            obs_list = []
        else:
            obs_list = [current_inventory]

        if self.regular_lead_time > 0:
            if isinstance(past_regular_orders, list):
                order_obs = torch.cat(
                    past_regular_orders[-self.regular_lead_time :], dim=-1
                )
            else:
                order_obs = past_regular_orders[:, -self.regular_lead_time :]

            if self.compressed:
                order_obs[0] += current_inventory

            obs_list.append(order_obs)

        if self.expedited_lead_time > 0:
            if isinstance(past_expedited_orders, list):
                order_obs = torch.cat(
                    past_expedited_orders[-self.expedited_lead_time :], dim=-1
                )
            else:
                order_obs = past_expedited_orders[:, -self.expedited_lead_time :]
            obs_list.append(order_obs)

        h = torch.cat(obs_list, dim=-1)
        h = self.stack(h)
        h = h - torch.frac(h).clone().detach()
        regular_q = h[:, 0].unsqueeze(-1)
        expedited_q = h[:, 1].unsqueeze(-1)
        return regular_q, expedited_q

    def get_total_cost(self, sourcing_model, sourcing_periods, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        total_cost = 0
        for i in range(sourcing_periods):
            current_inventory = sourcing_model.get_current_inventory()
            past_regular_orders = sourcing_model.get_past_regular_orders()
            past_expedited_orders = sourcing_model.get_past_expedited_orders()
            regular_q, expedited_q = self.forward(
                current_inventory, past_regular_orders, past_expedited_orders
            )
            sourcing_model.order(regular_q, expedited_q)
            current_cost = sourcing_model.get_cost(regular_q, expedited_q)
            total_cost += current_cost.mean()
        return total_cost

    def train(
        self,
        sourcing_model,
        sourcing_periods,
        epochs,
        validation_sourcing_periods=None,
        lr_init_inventory=1e-1,
        lr_parameters=3e-3,
        seed=None,
        tensorboard_writer=None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
        regular_lead_time = sourcing_model.get_regular_lead_time()
        expedited_lead_time = sourcing_model.get_expedited_lead_time()
        self.init_layers(regular_lead_time, expedited_lead_time)

        optimizer_init_inventory = torch.optim.RMSprop(
            [sourcing_model.init_inventory], lr=lr_init_inventory
        )
        optimizer_parameters = torch.optim.RMSprop(self.parameters(), lr=lr_parameters)
        min_cost = np.inf

        for epoch in range(epochs):
            # Clear grad cache
            optimizer_init_inventory.zero_grad()
            optimizer_parameters.zero_grad()
            # Reset the sourcing model with the learned init inventory
            sourcing_model.reset()
            total_cost = self.get_total_cost(sourcing_model, sourcing_periods)
            total_cost.backward()
            # Perform gradient descend
            if epoch % 3 == 0:
                optimizer_init_inventory.step()
            else:
                optimizer_parameters.step()
            # Save the best model
            if validation_sourcing_periods is not None and epoch % 10 == 0:
                eval_cost = self.get_total_cost(
                    sourcing_model, validation_sourcing_periods
                )
                if eval_cost < min_cost:
                    min_cost = eval_cost
                    best_state = self.state_dict()
            else:
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_state = self.state_dict()
            # Log train loss
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar(
                    "Avg. cost per period/train", total_cost / sourcing_periods, epoch
                )
                # Log evaluation loss
                tensorboard_writer.add_scalar(
                    "Avg. cost per period/eval",
                    eval_cost / validation_sourcing_periods,
                    epoch,
                )
                tensorboard_writer.flush()

        self.load_state_dict(best_state)

    def simulate(self, sourcing_model, sourcing_periods, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        sourcing_model.reset(batch_size=1)
        for i in range(sourcing_periods):
            current_inventory = sourcing_model.get_current_inventory()
            past_regular_orders = sourcing_model.get_past_regular_orders()
            past_expedited_orders = sourcing_model.get_past_expedited_orders()
            regular_q, expedited_q = self.forward(
                current_inventory, past_regular_orders, past_expedited_orders
            )
            sourcing_model.order(regular_q, expedited_q)
        past_inventories = sourcing_model.get_past_inventories()
        past_inventories = list(map(int, past_inventories))
        past_regular_orders = sourcing_model.get_past_regular_orders()
        past_regular_orders = list(map(int, past_inventories))
        past_expedited_orders = sourcing_model.get_past_expedited_orders()
        past_expedited_orders = list(map(int, past_expedited_orders))
        return past_inventories, past_regular_orders, past_expedited_orders

    def plot(self, sourcing_model, sourcing_periods):
        past_inventories, past_regular_orders, past_expedited_orders = self.simulate(
            sourcing_model=sourcing_model, sourcing_periods=sourcing_periods
        )
        fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
        ax[0].step(range(sourcing_periods), past_inventories[-sourcing_periods:])
        ax[0].yaxis.get_major_locator().set_params(integer=True)
        ax[0].set_title("Inventory")
        ax[0].set_xlabel("Period")
        ax[0].set_ylabel("Quantity")

        ax[1].step(
            range(sourcing_periods),
            past_expedited_orders[-sourcing_periods:],
            label="Expedited Order",
        )
        ax[1].step(
            range(sourcing_periods),
            past_regular_orders[-sourcing_periods:],
            label="Regular Order",
        )
        ax[1].yaxis.get_major_locator().set_params(integer=True)
        ax[1].set_title("Order")
        ax[1].set_xlabel("Period")
        ax[1].set_ylabel("Quantity")
        ax[1].legend()


class CappedDualIndexController:
    def __init__(self, s_e=0, s_r=0, q_r=0):
        """
        Parameters
        ----------
        s_e: int
            Capped dual index parameter 1
        s_r: int
            Capped dual index parameter 2
        q_r: int
            Capped dual index parameter 3

        Notes
        -----
        The function follows the implemetation of Sun, J., & Van Mieghem, J. A. (2019)([1]_).

        .. [1] Robust dual sourcing inventory management: Optimality of capped dual index policies and smoothing.
        Manufacturing & Service Operations Management, 21(4), 912-931.
        """
        self.s_e = s_e
        self.s_r = s_r
        self.q_r = q_r

    def capped_dual_index_sum(
        self,
        current_inventory,
        past_regular_orders,
        past_expedited_orders,
        regular_lead_time,
        expedited_lead_time,
        k,
    ):
        """
        Implementation of Eq. (3) of Sun, J., & Van Mieghem, J. A. (2019)([1]_).

        
        """
        inventory_position = current_inventory + sum(
            past_regular_orders[-regular_lead_time + i] for i in range(k + 1)
        )
        if expedited_lead_time > max(1, expedited_lead_time - k):
            inventory_position += sum(
                past_expedited_orders[-expedited_lead_time + i]
                for i in range(min(k, expedited_lead_time - 1) + 1)
            )
        return int(inventory_position)

    def forward(
        self,
        current_inventory,
        past_regular_orders,
        past_expedited_orders,
        regular_lead_time,
        expedited_lead_time,
    ):
        inventory_position = self.capped_dual_index_sum(
            current_inventory,
            past_regular_orders,
            past_expedited_orders,
            regular_lead_time,
            expedited_lead_time,
            k=0,
        )
        inventory_position_limit = self.capped_dual_index_sum(
            current_inventory,
            past_regular_orders,
            past_expedited_orders,
            regular_lead_time,
            expedited_lead_time,
            k=regular_lead_time - expedited_lead_time - 1,
        )
        regular_q = min(max(0, self.s_r - inventory_position_limit), self.q_r)
        expedited_q = max(0, self.s_e - inventory_position)
        return regular_q, expedited_q

    def get_total_cost(self, sourcing_model, sourcing_periods, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        regular_lead_time = sourcing_model.get_regular_lead_time()
        expedited_lead_time = sourcing_model.get_expedited_lead_time()
        total_cost = 0
        for i in range(sourcing_periods):
            current_inventory = sourcing_model.get_current_inventory()
            past_regular_orders = sourcing_model.get_past_regular_orders()
            past_expedited_orders = sourcing_model.get_past_expedited_orders()
            regular_q, expedited_q = self.forward(
                current_inventory,
                past_regular_orders,
                past_expedited_orders,
                regular_lead_time,
                expedited_lead_time,
            )
            sourcing_model.order(regular_q, expedited_q)
            current_cost = sourcing_model.get_cost(regular_q, expedited_q)
            total_cost += current_cost.mean()
        return total_cost

    def train(
        self,
        sourcing_model,
        sourcing_periods,
        s_e_range=np.arange(2, 11),
        s_r_range=np.arange(2, 11),
        q_r_range=np.arange(2, 11),
        seed=None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
        min_cost = np.inf
        for s_e in s_e_range:
            for s_r in s_r_range:
                for q_r in q_r_range:
                    sourcing_model.reset()
                    self.s_e = s_e
                    self.s_r = s_r
                    self.q_r = q_r
                    total_cost = self.get_total_cost(sourcing_model, sourcing_periods)
                    if total_cost < min_cost:
                        min_cost = total_cost
                        s_e_optimal = s_e
                        s_r_optimal = s_r
                        q_r_optimal = q_r
        self.s_e = s_e_optimal
        self.s_r = s_r_optimal
        self.q_r = q_r_optimal

    def simulate(self, sourcing_model, sourcing_periods, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        sourcing_model.reset()
        regular_lead_time = sourcing_model.get_regular_lead_time()
        expedited_lead_time = sourcing_model.get_expedited_lead_time()
        for i in range(sourcing_periods):
            current_inventory = sourcing_model.get_current_inventory()
            past_regular_orders = sourcing_model.get_past_regular_orders()
            past_expedited_orders = sourcing_model.get_past_expedited_orders()
            regular_q, expedited_q = self.forward(
                current_inventory,
                past_regular_orders,
                past_expedited_orders,
                regular_lead_time,
                expedited_lead_time,
            )
            sourcing_model.order(regular_q, expedited_q)
        past_inventories = sourcing_model.get_past_inventories()
        past_inventories = list(map(int, past_inventories))
        past_regular_orders = sourcing_model.get_past_regular_orders()
        past_regular_orders = list(map(int, past_inventories))
        past_expedited_orders = sourcing_model.get_past_expedited_orders()
        past_expedited_orders = list(map(int, past_expedited_orders))
        return past_inventories, past_regular_orders, past_expedited_orders

    def plot(self, sourcing_model, sourcing_periods):
        past_inventories, past_regular_orders, past_expedited_orders = self.simulate(
            sourcing_model=sourcing_model, sourcing_periods=sourcing_periods
        )
        fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
        ax[0].step(range(sourcing_periods), past_inventories[-sourcing_periods:])
        ax[0].yaxis.get_major_locator().set_params(integer=True)
        ax[0].set_title("Inventory")
        ax[0].set_xlabel("Period")
        ax[0].set_ylabel("Quantity")

        ax[1].step(
            range(sourcing_periods),
            past_expedited_orders[-sourcing_periods:],
            label="Expedited Order",
        )
        ax[1].step(
            range(sourcing_periods),
            past_regular_orders[-sourcing_periods:],
            label="Regular Order",
        )
        ax[1].yaxis.get_major_locator().set_params(integer=True)
        ax[1].set_title("Order")
        ax[1].set_xlabel("Period")
        ax[1].set_ylabel("Quantity")
        ax[1].legend()

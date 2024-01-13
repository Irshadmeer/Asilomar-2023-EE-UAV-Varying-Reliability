import numpy as np
import gym
from gym import spaces

from reliability import calculate_outage_probabilities
from movement import step_sde


class BasicScenario(gym.Env):
    def __init__(
        self,
        num_users,
        num_base_stations,
        max_coordinates,
        max_power_db,
        reliability_constraint,
        freq,
        rec_sensitivity,
        bs_locations=None,
        max_time=3000,
    ):
        self.num_users = num_users  # Number of users
        self.num_base_stations = num_base_stations  # Number of base stations
        self.max_coordinates = max_coordinates  # Maximum values in (x, y, z) directions
        self.max_power_db = max_power_db
        self.reliability_constraint = reliability_constraint
        self.bs_locations = None
        self.freq = freq
        self.rec_sensitivity = 10 ** (rec_sensitivity / 10)

        if bs_locations is None:
            bs_locations = (
                np.random.rand(self.num_base_stations, 3) * self.max_coordinates
            )
        assert np.shape(bs_locations) == (num_base_stations, 3)
        self.bs_locations = bs_locations

        self.observation_space = spaces.Dict(
            {
                "user_locations": spaces.Box(
                    low=0, high=np.inf, shape=(num_users, 3), dtype=float
                ),
                "user_velocities": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(num_users, 3), dtype=float
                ),
                "los": spaces.MultiBinary([num_users, num_base_stations]),
            }
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(num_users, num_base_stations), dtype=np.float32
        )

        self.state = {k: None for k in self.observation_space}
        self.time_slot = 0
        self.max_time = max_time

        self.movement_state = [
            {"x": [0, 0, 0], "y": [0, 0, 0]} for _ in range(self.num_users)
        ]

    def reset(self):
        self.time_slot = 0
        self.movement_state = [
            {"x": [0, 0, 0], "y": [0, 0, 0]} for _ in range(self.num_users)
        ]
        user_locations = np.tile(
            np.array(self.max_coordinates) / 2, (self.num_users, 1)
        )
        user_locations[:, 2] = (
            0.5 + 0.5 * np.random.rand(self.num_users)
        ) * self.max_coordinates[2]
        user_velocities = np.zeros((self.num_users, 3))  # Velocity vector (x, y, z)
        los = np.random.randint(0, 2, size=(self.num_users, self.num_base_stations))
        state_space = {
            "user_locations": user_locations,
            "user_velocities": user_velocities,
            "los": los,
        }
        self.state = state_space
        return state_space

    def _get_new_state(self):
        old_locations = self.state["user_locations"]
        new_locations = np.zeros_like(old_locations)
        new_locations[:, 2] = old_locations[:, 2]  # keep z component

        scale = 1000
        for user in range(self.num_users):
            _current_state_x = self.movement_state[user]["x"]
            _current_state_y = self.movement_state[user]["y"]
            _new_state_x = step_sde(_current_state_x, self.time_slot, dt=10 / scale)
            _new_state_y = step_sde(_current_state_y, self.time_slot, dt=10 / scale)
            self.movement_state[user]["x"] = _new_state_x
            self.movement_state[user]["y"] = _new_state_y
            new_locations[user, 0] = (
                scale * _new_state_x[0] + self.max_coordinates[0] / 2
            )
            new_locations[user, 1] = (
                scale * _new_state_y[0] + self.max_coordinates[1] / 2
            )
        new_velocities = np.abs(new_locations - old_locations)
        new_locations = np.array(new_locations, dtype=float)

        los_change = np.random.choice(
            [0, 1], size=(self.num_users, self.num_base_stations), p=[0.9, 0.1]
        )
        los = np.mod(self.state["los"] + los_change, 2)
        new_state = {
            "user_locations": new_locations,
            "user_velocities": new_velocities,
            "los": los,
        }
        self.time_slot = self.time_slot + 1
        return new_state

    def _get_outage_constraint(self):
        user_locations = self.state["user_locations"]
        _users_in_box_x = np.logical_and(
            self.max_coordinates[0] / 2 < user_locations[:, 0],
            user_locations[:, 0] < 2 * self.max_coordinates[0] / 3,
        )
        _users_in_box_y = np.logical_and(
            self.max_coordinates[1] / 2 < user_locations[:, 1],
            user_locations[:, 1] < 2 * self.max_coordinates[1] / 3,
        )
        _users_in_box = np.logical_and(_users_in_box_x, _users_in_box_y)
        outage_constraints = np.where(
            _users_in_box,
            self.reliability_constraint
            / 100,  # change here for the varying reliability constraints.
            self.reliability_constraint,
        )
        return outage_constraints, _users_in_box

    def step(self, action):
        user_locations = self.state["user_locations"]
        base_station_powers = 10 ** (
            self.max_power_db * action / 10.0
        )  # convert dB power (action) in linear scale
        los = self.state["los"]
        comp_power = np.ones((self.num_users, self.num_base_stations))
        max_power = 10 ** (self.max_power_db * comp_power / 10.0)
        outage_probabilities = calculate_outage_probabilities(
            user_locations,
            self.bs_locations,
            base_station_powers,
            self.freq,
            self.rec_sensitivity,
            los,
        )

        total_power = np.sum(base_station_powers)

        total_power_per_user = np.sum(base_station_powers, axis=1) / np.sum(
            max_power, axis=1
        )
        total_power = np.sum(base_station_powers)
        power_fraction = total_power / np.sum(max_power)

        power_fraction = total_power / np.sum(max_power)
        active_num_bs = np.zeros(
            self.num_users,
        )
        active_bs_indices = np.empty(len(action), dtype=object)
        for i, row in enumerate(action):
            active_num_bs[i] = len([x for x in row if x > 0])
            active_bs_indices[i] = [j for j, x in enumerate(row) if x > 0]

        num_active_connections = np.sum(active_num_bs)

        eps_max, users_in_box = self._get_outage_constraint()
        temp = eps_max - outage_probabilities
        reward_reliability = len([x for x in temp if x < 0]) / self.num_users
        reward_ee = 1 - total_power / np.sum(max_power)
        reward_active_bs = 1 - (
            num_active_connections / (self.num_users * self.num_base_stations)
        )

        reward = reward_ee + reward_active_bs - reward_reliability

        # info = self._get_info()
        info = {
            "time_slot": self.time_slot,
            "action": action,
            "total_power": total_power,
            "total_power_per_user": total_power_per_user,
            "energy_efficiency": 1.0 / total_power,
            "reward": reward,
            "reward_ee": reward_ee,
            "reward_reliability": reward_reliability,
            "outage_probabilities": outage_probabilities,
            "outage_violations": np.count_nonzero(outage_probabilities > eps_max),
            "power_fraction": power_fraction,
            "num_active_BS": active_num_bs[0],
            "bs_locations": self.bs_locations,
            "user_locations": user_locations,
            "active_bs_index": active_bs_indices,
            "user_in_high_rel_zone": users_in_box,
        }

        info.update(self.state)
        new_state = self._get_new_state()
        self.state = new_state

        done = self.is_done()
        return new_state, reward, done, info

    def is_done(self):
        if self.time_slot > self.max_time:
            return True
        new_user_locations = self.state["user_locations"]
        if np.any(new_user_locations < 0) or np.any(
            new_user_locations > self.max_coordinates
        ):
            return True
        return False


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    num_uavs = 1
    num_bs = 4
    max_x = 500
    max_y = 500
    max_z = 50
    max_power_db = 20
    reliability_constraint = 0.1
    carrier_freq = 2.4  # GHz
    rec_sensitivity = -100  # dBm

    env = BasicScenario(
        num_uavs,
        num_bs,
        [max_x, max_y, max_z],
        max_power_db,
        reliability_constraint,
        carrier_freq,
        rec_sensitivity,
    )
    # It will check your custom environment and output additional warnings if needed
    check_env(env)

import numpy as np


def power_alo_for_closest_bs(observation: dict, bs_locs: np.array):
    user_locs = observation["user_locations"]
    user_locs = np.array(user_locs)
    bs_locs = np.array(bs_locs)
    # Calculate the Euclidean distance between each user location and each basestation location
    dists = np.sqrt(np.sum((user_locs[:, np.newaxis] - bs_locs) ** 2, axis=2))
    # Find the index of the closest basestation for each user
    closest_bs_indices = np.argmin(dists, axis=1)
    bs_mask = np.where(
        np.arange(len(bs_locs)) == closest_bs_indices[:, np.newaxis], 1, 0
    )
    # Return power
    power_allocation = bs_mask
    return power_allocation, None


def power_alo_from_all_bs(observation: dict, n_users: int, n_base_stations: int):
    return np.ones((n_users, n_base_stations)), None


if __name__ == "__main__":
    uav_locations = [
        [10, 10, 50],
        [50, 50, 10],
        [80, 80, 30],
        [250, 125, 15],
        [327, 234, 60],
    ]
    bs_locations = [[100, 100, 50], [100, 400, 25], [400, 400, 15], [400, 100, 30]]
    max_power_db = 30
    bs_powers = power_alo_for_closest_bs(uav_locations, bs_locations)
    print(bs_powers)

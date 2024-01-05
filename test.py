import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, SAC, PPO, DDPG
import pickle
import random

from environment import calculate_outage_probabilities
from environment import BasicScenario
from baseline import power_alo_for_closest_bs,power_alo_from_all_bs
from util import get_algorithm_function



def main( model_path,total_episodes=1000, plot: bool = True, **kwargs):
    # Load the saved model from the file
    #model_path= ["SAC_model.zip"]
    bs_locations=np.array([(random.uniform(0, 3000), random.uniform(0, 3000), 25) for _ in range(kwargs['num_base_stations'])])
    env = BasicScenario(**kwargs, bs_locations=None)
    ALGORITHMS = {
                  "Closest": lambda x: power_alo_for_closest_bs(x, bs_locs=bs_locations),
                  "COMP": lambda x: power_alo_from_all_bs(x,n_users= kwargs['num_users'],n_base_stations=kwargs['num_base_stations']),
                 }

    for model_file in model_path:
        _modelname = os.path.basename(model_file)
        _algorithm = os.path.basename(model_file).split("_", 1)[0]
        func_algorithm = get_algorithm_function(_algorithm)
        model = func_algorithm.load(model_file)
        ALGORITHMS[_modelname] = (lambda x, model=model: model.predict(x, deterministic=True))

    test_data = {k: [[] for _ in range(total_episodes)] for k in ALGORITHMS}

    for _name, algorithm in ALGORITHMS.items():
        print(f"Working on algorithm: {_name}")
        for episode in range(total_episodes):
            print(f"Episode {episode+1:d}/{total_episodes:d}")
            np.random.seed(episode)
            env.seed(episode)
            obs = env.reset()
            done = False
            while not done:
                action, _states = algorithm(obs)
                obs, reward, done, info = env.step(action)
                test_data[_name][episode].append(info)
        file_name = f"test_data_{_name}.pickle"
        with open(file_name, 'wb') as f:
            pickle.dump(test_data, f)        

    
    return



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", nargs="+", default=["SAC_model.zip"])
    parser.add_argument("-n", "--num_users", type=int, default=3)
    parser.add_argument("-b", "--num_base_stations", type=int, default=19)
    parser.add_argument(
        "-x", "--max_coordinates", nargs=3, type=float, default=[3000, 3000, 120]
    )
    parser.add_argument("-p", "--max_power_db", type=float, default=24)
    parser.add_argument("-e", "--reliability_constraint", type=float, default=.00001)
    parser.add_argument("-f", "--freq", type=float, default=5.4)
    parser.add_argument("-s", "--rec_sensitivity", type=float, default=-100)
    parser.add_argument("-t", "--total_episodes", type=int, default=10000)
    #parser.add_argument("--plot", action="store_true")

    args = vars(parser.parse_args())
    main(**args)
    plt.show()

import os.path
import pickle
import numpy as np
import random

from stable_baselines3 import A2C, SAC, PPO, DDPG
from stable_baselines3 import common
import matplotlib.pyplot as plt

from environment import BasicScenario
from loggers import SummaryWriterCallback
from data_logger import StoreDataCallback
from util import get_algorithm_function
from rander import Render, render_animation


def main(algorithm, output, num_tries, max_coordinates, plot: bool = False, **kwargs):
    bs_locations=np.array([(random.uniform(0, 3000), random.uniform(0, 3000), 25) for _ in range(kwargs['num_base_stations'])])
    env = BasicScenario(bs_locations=None,max_coordinates=max_coordinates, **kwargs)
    env = common.monitor.Monitor(env, allow_early_resets=False)
    

    func_algorithm = get_algorithm_function(algorithm)
    model = func_algorithm("MultiInputPolicy", env, verbose=1,
                           tensorboard_log="./tensorboard_log/")
    tb_callback = SummaryWriterCallback()
    df_callback = StoreDataCallback()
    callbacks = [ df_callback, tb_callback]
    model.learn(total_timesteps=num_tries, progress_bar=True,
                callback=callbacks)
    #render_animation(info_list=df_callback.history_data,max_coordinates=max_coordinates)
    
    #save data in pickle
    file_name = f"simulation_data_{algorithm}.pickle"
    with open(file_name, 'wb') as f:
        pickle.dump(df_callback.history_data, f)

    output = os.path.join(os.path.dirname(output),
                          "{}_{}".format(algorithm, os.path.basename(output)))
    model.save(output)

    if plot:
        fix, ax1 = plt.subplots()
        ax1.plot(env.get_episode_lengths(), 'bo-')
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(env.get_episode_rewards(), 'ro-')
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", default="DDPG")
    parser.add_argument("-n", "--num_users", type=int, default=3)
    parser.add_argument("-b", "--num_base_stations", type=int, default=19)
    parser.add_argument("-x", "--max_coordinates", nargs=3, type=float, default=[3000, 3000, 120])
    parser.add_argument("-p", "--max_power_db", type=float, default=24)
    parser.add_argument("-e", "--reliability_constraint", type=float, default=.00001)
    parser.add_argument("-f", "--freq", type=float, default=2.4)
    parser.add_argument("-s", "--rec_sensitivity", type=float, default=-100)
    parser.add_argument("-o", "--output", default="model.zip")
    parser.add_argument("-t", "--num_tries", type=int, default=600000)
    parser.add_argument("--plot", action="store_true")

    args = vars(parser.parse_args())
    model = main(**args)
    #plt.show()

from stable_baselines3 import A2C, SAC, PPO, DDPG


def get_algorithm_function(algorithm):
    algorithm = algorithm.lower()
    if algorithm == "sac":
        func_algorithm = SAC
    elif algorithm == "ppo":
        func_algorithm = PPO
    elif algorithm == "ddpg":
        func_algorithm = DDPG
    elif algorithm == "a2c":
        func_algorithm = A2C
    else:
        raise NotImplementedError(f"Algorithm {algorithm} is not implemented")
    return func_algorithm

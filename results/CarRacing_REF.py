import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

from agents.actor_critic_agents.SAC import SAC
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config

import chainerrl
from envs.common_envs_utils.env_wrappers import *
from envs.gym_car_intersect_fixed import CarRacingHackatonContinuousFixed

env = CarRacingHackatonContinuousFixed()
env = chainerrl.wrappers.ContinuingTimeLimit(env, max_episode_steps=250)
env = MaxAndSkipEnv(env, skip=4)
env = WarpFrame(env, channel_order='chw')
env._max_episode_steps = 250


config = Config()
config.seed = 1
config.environment = env
config.num_episodes_to_run = 450
config.file_to_save_data_results = 'result_cars'
config.file_to_save_results_graph = 'graph_cars'
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = False
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = True


config.hyperparameters = {

    "Actor_Critic_Agents": {
            "Actor": {
                "learning_rate": 0.003,
                "linear_hidden_units": [20, 20],
                "final_layer_activation": None,
                "batch_norm": False,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "initialiser": "Xavier"
            },

            "Critic": {
                "learning_rate": 0.02,
                "linear_hidden_units": [20, 20],
                "final_layer_activation": None,
                "batch_norm": False,
                "buffer_size": 1000000,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "initialiser": "Xavier"
            },

        "min_steps_before_learning": 1000, #for SAC only
        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0,  # for O-H noise
        "theta": 0.15,  # for O-H noise
        "sigma": 0.25,  # for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 20,
        "learning_updates_per_learning_session": 10,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": True,
        "do_evaluation_iterations": True,
        "clip_rewards": False

    }

}

if __name__ == "__main__":
    AGENTS = [SAC]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()




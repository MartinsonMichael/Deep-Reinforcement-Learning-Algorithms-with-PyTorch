import argparse
import os
import sys
from os.path import dirname, abspath

sys.path.append(dirname(dirname(abspath(__file__))))

from agents.actor_critic_agents.SAC import SAC
from utilities.data_structures.Config import Config

import chainerrl
import tensorflow as tf
from envs.common_envs_utils.extended_env_wrappers import ExtendedMaxAndSkipEnv, FrameCompressor, OriginalStateKeeper, \
    ImageWithVectorCombiner, ChannelSwapper, TorchTensorCaster
from envs.gym_car_intersect_fixed import CarRacingHackatonContinuousFixed


def create_env_both(settings_path=None):
    env = CarRacingHackatonContinuousFixed(settings_file_path=settings_path)
    # -> dict[(.., .., 3), (16)]
    env = chainerrl.wrappers.ContinuingTimeLimit(env, max_episode_steps=250)
    env = ExtendedMaxAndSkipEnv(env, skip=4)
    env = FrameCompressor(env)
    # -> dict[(84, 84, 3), (16)]
    # env = OriginalStateKeeper(env, 'uncombined_state')
    env = ImageWithVectorCombiner(env)
    # -> Box(84, 84, 19)
    env = ChannelSwapper(env)
    # -> Box(19, 84, 84)
    env = TorchTensorCaster(env)
    env._max_episode_steps = 250
    return env


def create_config(args):
    config = Config()
    config.seed = 1
    config.environment = None
    if args.mode == 'both':
        config.environment = create_env_both(args.env_settings)
    else:
        raise NotImplemented

    config.num_episodes_to_run = 1500
    config.file_to_save_data_results = 'result_cars'
    config.file_to_save_results_graph = 'graph_cars'
    config.show_solution_score = True
    config.visualise_individual_results = False
    config.visualise_overall_agent_results = True
    config.standard_deviation_results = 1.0
    config.runs_per_agent = 1
    config.device = args.device
    config.overwrite_existing_results_file = False
    config.randomise_random_seed = True
    config.save_model = True
    config.max_episode_steps = 300

    config.hyperparameters = {
        "Actor_Critic_Agents": {
            "Actor": {
                "learning_rate": 3e-4,
                "linear_hidden_units": [20, 20],
                "final_layer_activation": None,
                "batch_norm": False,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "initialiser": "Xavier"
            },
            "Critic": {
                "learning_rate": 3e-4,
                "linear_hidden_units": [20, 20],
                "final_layer_activation": None,
                "batch_norm": False,
                "buffer_size": 200000,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "initialiser": "Xavier"
            },
            "min_steps_before_learning": 500,
            "batch_size": 128,
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
            "do_evaluation_iterations": False,
            "clip_rewards": False
        }
    }
    return config


def main(args):
    agent_title = args.name
    if not os.path.exists(os.path.join('logs', agent_title)):
        os.makedirs(os.path.join('logs', agent_title))
    tf_writer = tf.summary.create_file_writer(os.path.join('logs', agent_title))
    agent_config = create_config(args)
    agent_config.name = agent_title

    # random.randint(0, 2 ** 32 - 2)
    agent_config.seed = 42

    agent_config.hyperparameters = agent_config.hyperparameters['Actor_Critic_Agents']
    print("AGENT NAME: {}".format('SAC'))

    agent = SAC(agent_config)

    print(agent.hyperparameters)

    print("RANDOM SEED ", agent_config.seed)

    game_scores, rolling_scores, time_taken = agent.run_n_episodes(tf_saver=tf_writer, visualize=True)
    print("Time taken: {}".format(time_taken), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='test', help='name for experiment')
    parser.add_argument('--mode', type=str, default='both', help='image only, vector only, or their combination')
    parser.add_argument('--env-settings', type=str, default='test', help='path to CarRacing env settings')
    parser.add_argument('--device', type=str, default='cpu', help='path to CarRacing env settings')
    args = parser.parse_args()

    if args.mode not in ['image', 'vector', 'both']:
        raise ValueError("mode should be one of 'image', 'vector', 'both'")
    main(args)

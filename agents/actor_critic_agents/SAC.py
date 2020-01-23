import time

from agents.Base_Agent import Base_Agent
from agents.actor_critic_agents.utils_continues import QNet, Policy
from envs.common_envs_utils.env_state_utils import from_combined_state_to_image_vector, \
    from_image_vector_to_combined_state
from utilities.OU_Noise import OU_Noise
from torch.optim import Adam
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import tensorflow as tf

from utilities.data_structures.Torch_Replay_Buffer import Torch_Replay_Buffer
from utilities.data_structures.Torch_Separated_Replay_Buffer import Torch_Separated_Replay_Buffer

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
TRAINING_EPISODES_PER_EVAL_EPISODE = 10
EPSILON = 1e-6


class SAC(Base_Agent):
    """Soft Actor-Critic model based on the 2018 paper https://arxiv.org/abs/1812.05905 and on this github implementation
      https://github.com/pranz24/pytorch-soft-actor-critic. It is an actor-critic algorithm where the agent is also trained
      to maximise the entropy of their actions as well as their cumulative reward"""
    agent_name = "SAC"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        assert self.action_types == "CONTINUOUS", "Action types must be continuous. Use SAC Discrete instead for " \
                                                  "discrete actions "
        assert self.config.hyperparameters["Actor"]["final_layer_activation"] != "Softmax", "Final actor layer must " \
                                                                                            "not be softmax "
        self.hyperparameters = config.hyperparameters

        self.critic_local = QNet(
            state_description=self.config.environment.observation_space,
            action_size=self.action_size,
            hidden_size=256,
            device=self.device,
        )
        self.critic_local_2 = QNet(
            state_description=self.config.environment.observation_space,
            action_size=self.action_size,
            hidden_size=256,
            device=self.device,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic_local.parameters(),
            lr=self.hyperparameters["Critic"]["learning_rate"],
            eps=1e-4,
        )
        self.critic_optimizer_2 = torch.optim.Adam(
            self.critic_local_2.parameters(),
            lr=self.hyperparameters["Critic"]["learning_rate"],
            eps=1e-4,
        )

        self.critic_target = QNet(
            state_description=self.config.environment.observation_space,
            action_size=self.action_size,
            hidden_size=256,
            device=self.device,
        )
        self.critic_target_2 = QNet(
            state_description=self.config.environment.observation_space,
            action_size=self.action_size,
            hidden_size=256,
            device=self.device,
        )
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)

        self.memory = Torch_Replay_Buffer(
            self.hyperparameters["Critic"]["buffer_size"],
            self.hyperparameters["batch_size"],
            self.config.seed,
            device=self.device,
        )

        # self.memory = Torch_Separated_Replay_Buffer(
        #     self.hyperparameters["Critic"]["buffer_size"],
        #     self.hyperparameters["batch_size"],
        #     self.config.seed,
        #     device=self.device,
        #     state_extractor=from_combined_state_to_image_vector,
        #     state_producer=from_image_vector_to_combined_state,
        # )

        self.actor_local = Policy(
            state_description=self.config.environment.observation_space,
            action_size=self.action_size,
            hidden_size=256,
            device=self.device,
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor_local.parameters(),
            lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4
        )
        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.environment.action_space.shape).to(
                self.device)).item()  # heuristic value from the paper
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]

        self.add_extra_noise = self.hyperparameters["add_extra_noise"]
        if self.add_extra_noise:
            self.noise = OU_Noise(self.action_size, self.config.seed, self.hyperparameters["mu"],
                                  self.hyperparameters["theta"], self.hyperparameters["sigma"])

        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]
        self._game_stats = {}

    def save_result(self):
        """Saves the result of an episode of the game. Overriding the method in Base Agent that does this because we only
        want to keep track of the results during the evaluation episodes"""
        if self.episode_number == 1 or not self.do_evaluation_iterations:
            self.game_full_episode_scores.extend([self.total_episode_score_so_far])
            self.rolling_results.append(np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]))
            self.save_max_result_seen()

        elif (self.episode_number - 1) % TRAINING_EPISODES_PER_EVAL_EPISODE == 0:
            self.game_full_episode_scores.extend(
                [self.total_episode_score_so_far for _ in range(TRAINING_EPISODES_PER_EVAL_EPISODE)])
            self.rolling_results.extend(
                [np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]) for _ in
                 range(TRAINING_EPISODES_PER_EVAL_EPISODE)])
            self.save_max_result_seen()

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        # np.random.seed(int(round(time.time() * 1000)) % 2**31)
        Base_Agent.reset_game(self)
        if self.add_extra_noise:
            self.noise.reset()
        self._game_stats = {}

    def run_n_episodes(
            self,
            num_episodes=None,
            show_whether_achieved_goal=True,
            save_and_print_results=True,
            tf_saver=None,
            visualize=True,
    ):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        if num_episodes is None:
            num_episodes = self.config.num_episodes_to_run
        start = time.time()
        while self.episode_number < num_episodes:
            self.reset_game()

            need_visualize = visualize if self.episode_number % 5 == 0 else False

            if 'mode_to_use' not in self.hyperparameters.keys():
                self.step(visualize=need_visualize)
            else:
                if self.hyperparameters['mode_to_use'] == 'rlkit':
                    self.rlkit_step(visualize=need_visualize)
                else:
                    self.step(visualize=need_visualize)


            if save_and_print_results:
                self.save_and_print_result()
            if tf_saver is not None:
                self.create_tf_charts(tf_saver)

        time_taken = time.time() - start
        if show_whether_achieved_goal:
            self.show_whether_achieved_goal()
        # if self.config.save_model:
        #     self.locally_save_policy()
        return self.game_full_episode_scores, self.rolling_results, time_taken

    def rlkit_step(self, visualize=False):
        """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
        eval_ep = self.episode_number % TRAINING_EPISODES_PER_EVAL_EPISODE == 0 and self.do_evaluation_iterations
        self.episode_step_number_val = 0
        was_done = False
        while not self.done:
            if was_done:
                self.reset_game()

            self.episode_step_number_val += 1
            self.action = self.pick_action(eval_ep)
            self.conduct_action(self.action)

            mask = self.done
            if not eval_ep:
                self.memory.add_experience(
                    self.state, self.action, self.reward, self.next_state, mask
                )
            self.state = self.next_state
            self.global_step_number += 1
            print(f'global steps : {self.global_step_number}')

            if self.episode_step_number_val >= self.hyperparameters['rlkit_mode_parameters']['explanation_steps_per_step']:
                break

            if self.episode_step_number_val > self.config.max_episode_steps + 10:
                was_done = True

        if self.global_step_number > self.hyperparameters["min_steps_before_learning"] and \
                self.enough_experiences_to_learn_from():

            for _ in range(self.hyperparameters['rlkit_mode_parameters']['update_steps_per_step']):
                self.learn()

        if eval_ep:
            self.print_summary_of_latest_evaluation_episode()
        self.episode_number += 1

        if visualize and self.global_step_number > self.hyperparameters["min_steps_before_learning"]:
            from envs.common_envs_utils import episode_visualizer
            episode_visualizer(
                env=self.environment,
                action_picker=lambda state: self.actor_pick_action(state, eval=True),
                name=self.config.name,
            )

    def step(self, visualize=False):
        """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
        eval_ep = self.episode_number % TRAINING_EPISODES_PER_EVAL_EPISODE == 0 and self.do_evaluation_iterations
        self.episode_step_number_val = 0
        while not self.done:
            self.episode_step_number_val += 1
            self.action = self.pick_action(eval_ep)
            self.conduct_action(self.action)

            if self.time_for_critic_and_actor_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    self.learn()
            # mask = False if self.episode_step_number_val >= self.config.max_episode_steps else self.done
            mask = self.done
            if not eval_ep:
                # self.save_experience(experience=(self.state, self.action, self.reward, self.next_state, mask))
                self.memory.add_experience(
                    self.state, self.action, self.reward, self.next_state, mask
                )
            self.state = self.next_state
            self.global_step_number += 1
            print(f'global steps : {self.global_step_number}')

            if self.episode_step_number_val > self.config.max_episode_steps + 10:
                break

        print(self.total_episode_score_so_far)
        if eval_ep:
            self.print_summary_of_latest_evaluation_episode()
        self.episode_number += 1

        if visualize and self.global_step_number > self.hyperparameters["min_steps_before_learning"]:
            from envs.common_envs_utils import episode_visualizer
            episode_visualizer(
                env=self.environment,
                action_picker=lambda state: self.actor_pick_action(state, eval=True),
                name=self.config.name,
            )

    def update_stats_due_to_step_info(self, info, reward, done):
        if not done:
            return
        if 'is_finish' in info.keys():
            if info['is_finish']:
                self._game_stats['finished'] = 1
            else:
                self._game_stats['finished'] = 0
        if 'time' in info.keys():
            self._game_stats['env steps taken'] = info['time']
        if 'track_progress' in info.keys():
            self._game_stats['track_progress'] = info['track_progress']

        self._game_stats['temperature'] = self.alpha.cpu().detach().numpy()[0]

    def create_tf_charts(self, tf_writer):
        if self.global_step_number < self.hyperparameters['min_steps_before_learning']:
            return
        with tf_writer.as_default():
            tf.summary.scalar(
                name='rolling score',
                data=np.array(self.game_full_episode_scores[-7:]).mean(),
                step=self.episode_number
            )
            tf.summary.scalar(name='score', data=self.game_full_episode_scores[-1], step=self.episode_number)
            for name, value in self._game_stats.items():
                if 'loss' in name and self.episode_step_number_val > 0:
                    value /= self.episode_step_number_val
                    name = 'average ' + name
                tf.summary.scalar(name=name, data=value, step=self.episode_number)

    def pick_action(self, eval_ep, state=None):
        """Picks an action using one of three methods: 1) Randomly if we haven't passed a certain number of steps,
         2) Using the actor in evaluation mode if eval_ep is True  3) Using the actor in training mode if eval_ep is False.
         The difference between evaluation and training mode is that training mode does more exploration"""
        if state is None:
            state = self.state
        if eval_ep:
            action = self.actor_pick_action(state=state, eval=True)
        elif self.global_step_number < self.hyperparameters["min_steps_before_learning"]:
            action = np.random.uniform(-1, 1, self.environment.action_space.shape)
            print("Picking random action ", action)
        else:
            action = self.actor_pick_action(state=state)
        if self.add_extra_noise:
            action += self.noise.sample()
        return action

    def actor_pick_action(self, state=None, eval=False):
        """Uses actor to pick an action in one of two ways: 1) If eval = False and we aren't in eval mode then it picks
        an action that has partly been randomly sampled 2) If eval = True then we pick the action that comes directly
        from the network and so did not involve any random sampling"""
        if state is None:
            state = self.state

        if not eval:
            action, _, _ = self.produce_action_and_action_info(state)
        else:
            with torch.no_grad():
                _, z, action = self.produce_action_and_action_info(state)
        action = action.detach().cpu().numpy()
        return action[0]

    def produce_action_and_action_info(self, state):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""
        actor_output = self.actor_local(state)
        mean, log_std = actor_output[:, :self.action_size], actor_output[:, self.action_size:]
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # rsample means it is sampled using reparameterisation trick
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

    def time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return (
                self.global_step_number > self.hyperparameters["min_steps_before_learning"] and
                self.enough_experiences_to_learn_from() and
                self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0
        )

    def learn(self):
        """Runs a learning iteration for the actor, both critics and (if specified) the temperature parameter"""
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.sample_experiences()
        qf1_loss, qf2_loss = self.calculate_critic_losses(state_batch, action_batch, reward_batch, next_state_batch,
                                                          mask_batch)
        policy_loss, log_pi = self.calculate_actor_loss(state_batch)
        if self.automatic_entropy_tuning:
            alpha_loss = self.calculate_entropy_tuning_loss(log_pi)
        else:
            alpha_loss = None
        self.update_all_parameters(qf1_loss, qf2_loss, policy_loss, alpha_loss)

    def sample_experiences(self):
        return self.memory.sample()

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.produce_action_and_action_info(next_state_batch)

            # print(f'next_state_batch : {next_state_batch.shape}')
            # print(f'next_state_action : {next_state_action.shape}')

            qf1_next_target = self.critic_target(next_state_batch, next_state_action)
            qf2_next_target = self.critic_target_2(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi

            # print(f'reward : {reward_batch.shape}')
            # print(f'mask : {mask_batch.shape}')
            # print(f'min_qf_next_target : {min_qf_next_target.shape}')

            next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * (
                min_qf_next_target)

            # print(f'next_q_value : {next_q_value.shape}')

        qf1 = self.critic_local(state_batch, action_batch)
        qf2 = self.critic_local_2(state_batch, action_batch)

        # print(f'state batch : {state_batch.shape}')
        # print(f'qf1 : {qf1.shape}')

        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, log_pi, _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local(state_batch, action)
        qf2_pi = self.critic_local_2(state_batch, action)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        return policy_loss, log_pi

    def calculate_entropy_tuning_loss(self, log_pi):
        """Calculates the loss for the entropy temperature parameter. This is only relevant if self.automatic_entropy_tuning
        is True."""
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss

    def update_all_parameters(self, critic_loss_1, critic_loss_2, actor_loss, alpha_loss):
        """Updates the parameters for the actor, both critics and (if specified) the temperature parameter"""

        # from torch.utils.tensorboard import SummaryWriter
        # writer = SummaryWriter('test_log_dir')
        # writer.add_graph(critic_loss_1)
        # writer.add_graph(actor_loss)

        for loss_name, loss_value in [
            ('q1 loss', critic_loss_1),
            ('q2 loss', critic_loss_2),
            ('policy loss', actor_loss),
            ('temperature loss', alpha_loss),
        ]:
            if loss_name not in self._game_stats.keys():
                self._game_stats[loss_name] = 0.0
            self._game_stats[loss_name] += loss_value.cpu().detach().numpy()

        self.take_optimisation_step(
            self.critic_optimizer,
            self.critic_local,
            critic_loss_1,
            self.hyperparameters["Critic"]["gradient_clipping_norm"]
        )
        self.take_optimisation_step(self.critic_optimizer_2, self.critic_local_2, critic_loss_2,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local, self.critic_target,
                                           self.hyperparameters["Critic"]["tau"])
        self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2,
                                           self.hyperparameters["Critic"]["tau"])
        if alpha_loss is not None:
            self.take_optimisation_step(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()

    def print_summary_of_latest_evaluation_episode(self):
        """Prints a summary of the latest episode"""
        print(" ")
        print("----------------------------")
        print("Episode score {} ".format(self.total_episode_score_so_far))
        print("----------------------------")

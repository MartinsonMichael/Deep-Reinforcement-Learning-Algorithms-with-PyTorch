from collections import namedtuple, deque
import random
import torch
import numpy as np
from enum import Enum


class ReplayBufferMode(Enum):
    only_pictures = 0,
    only_vectors = 1,
    both = 2,


class Extended_Replay_Buffer(object):
    """Replay buffer to store past experiences that the agent can then use for training data"""
    
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
            "state_picture",
            "state_extra"
            "action",
            "reward",
            "next_state_picture",
            "next_state_extra"
            "done"
        ])
        self.seed = random.seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add_experience(
        self,
        states,
        actions, rewards,
        next_states,
        dones
    ):
        """Adds experience(s) into the replay buffer"""

        states_picture, states_extra = None, None
        if 'picture' in states.keys():
            states_picture = states['picture']
        if 'vector' in states.keys():
            states_extra = states['vector']

        next_states_picture, next_states_extra = None, None
        if 'picture' in next_states.keys():
            next_states_picture = next_states['picture']
        if 'vector' in next_states.keys():
            next_states_extra = next_states['vector']

        if type(dones) == list:
            assert type(dones[0]) != list, "A done shouldn't be a list"
            experiences = [
                self.experience(state_picture, state_extra, action, reward, next_state_picture, next_state_extra, done)
                for state_picture, state_extra, action, reward, next_state_picture, next_state_extra, done in
                zip(states_picture, states_extra, actions, rewards, next_states_picture, next_states_extra, dones)
            ]
            self.memory.extend(experiences)
        else:
            experience = self.experience(
                states_picture,
                states_extra,
                actions,
                rewards,
                next_states_picture,
                next_states_extra,
                dones,
            )
            self.memory.append(experience)
   
    def sample(self, num_experiences=None, separate_out_data_types=True):
        """Draws a random sample of experience from the replay buffer"""
        experiences = self.pick_experiences(num_experiences)
        if separate_out_data_types:
            states_picture, states_extra, actions, rewards, next_states, next_states_extra, dones = self.separate_out_data_types(experiences)
            return (
                {'picture': states_picture, 'vector': states_extra},
                actions,
                rewards,
                {'picture': next_states, 'vector': next_states_extra},
                dones,
            )

        raise ValueError('ты не прав')
            
    def separate_out_data_types(self, experiences):
        """Puts the sampled experience into the correct format for a PyTorch neural network"""
        states_picture = torch.from_numpy(np.array([e.state_picture for e in experiences])).float().to(self.device)
        states_extra = torch.from_numpy(np.array([e.state_extra for e in experiences])).float().to(self.device)

        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(self.device)

        next_states_picture = torch.from_numpy(np.array([e.next_state_picture for e in experiences])).float().to(self.device)
        next_states_extra = torch.from_numpy(np.array([e.next_state_extra for e in experiences])).float().to(self.device)

        dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences])).float().to(self.device)
        
        return states_picture, states_extra, actions, rewards, next_states_picture, next_states_extra, dones
    
    def pick_experiences(self, num_experiences=None):
        if num_experiences is not None:
            batch_size = num_experiences
        else:
            batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)

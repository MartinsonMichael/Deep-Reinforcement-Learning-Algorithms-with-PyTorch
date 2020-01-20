from collections import namedtuple, deque
import random
import torch
import numpy as np


class Torch_Replay_Buffer(object):
    """Replay buffer to store past experiences that the agent can then use for training data"""
    
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add_experience(self, states, actions, rewards, next_states, dones):
        """Adds experience(s) into the replay buffer"""
        if type(dones) == list:
            assert type(dones[0]) != list, "A done shouldn't be a list"
            experiences = [self.experience(state, action, reward, next_state, done)
                           for state, action, reward, next_state, done in
                           zip(states, actions, rewards, next_states, dones)]
            self.memory.extend(experiences)
        else:
            experience = self.experience(states, actions, rewards, next_states, dones)
            self.memory.append(experience)
   
    def sample(self, num_experiences=None):
        """Draws a random sample of experience from the replay buffer"""
        experiences = self.pick_experiences(num_experiences)
        return (
            torch.from_numpy(np.array([e.state for e in experiences], dtype=np.float32)).to(self.device),
            torch.from_numpy(np.array([e.action for e in experiences], dtype=np.float32)).to(self.device),
            torch.from_numpy(np.array([e.reward for e in experiences], dtype=np.float32)).to(self.device),
            torch.from_numpy(np.array([e.next_state for e in experiences], dtype=np.float32)).to(self.device),
            torch.from_numpy(np.array([e.done for e in experiences], dtype=np.float32)).to(self.device),
        )

    def pick_experiences(self, num_experiences=None):
        if num_experiences is not None:
            batch_size = num_experiences
        else:
            batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)

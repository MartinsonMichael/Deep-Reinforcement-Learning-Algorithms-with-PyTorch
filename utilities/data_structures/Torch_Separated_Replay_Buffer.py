from collections import namedtuple, deque
import random
import torch
import numpy as np


class Torch_Separated_Replay_Buffer(object):
    """Replay buffer to store past experiences that the agent can then use for training data"""
    
    def __init__(self, buffer_size, batch_size, seed, device, state_extractor, state_producer):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
            "state_picture",
            "state_vector",
            "action",
            "reward",
            "next_state_picture",
            "next_state_vector",
            "done"
        ])
        self.seed = random.seed(seed)
        self.device = device
        self._state_extractor = state_extractor
        self._state_producer = state_producer

    def add_experience(self, states, actions, rewards, next_states, dones):
        """Adds experience(s) into the replay buffer"""
        if type(dones) == list:
            assert type(dones[0]) != list, "A done shouldn't be a list"
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                self._add_single_experience(state, action, reward, next_state, done)
        else:
            self._add_single_experience(states, actions, rewards, next_states, dones)

    def _add_single_experience(self, state, action, reward, next_state, done):
        state_picture, state_vector = self._state_extractor(state)
        next_state_picture, next_state_vector = self._state_extractor(next_state)

        if state_picture is not None:
            assert isinstance(state_picture, np.ndarray), "state_picture must be None or np.ndarray"
            assert isinstance(next_state_picture, np.ndarray), "state_picture must be None or np.ndarray"
            state_picture = state_picture.astype(np.uint8)
            next_state_picture = next_state_picture.astype(np.uint8)
        if state_vector is not None:
            assert isinstance(state_vector, np.ndarray), "state_vector must me None or np.ndarray"
            assert isinstance(next_state_vector, np.ndarray), "state_vector must me None or np.ndarray"

        self.memory.append(self.experience(
            state_picture,
            state_vector,
            action,
            reward,
            next_state_picture,
            next_state_vector,
            done
        ))
   
    def sample(self, num_experiences=None):
        """Draws a random sample of experience from the replay buffer"""
        experiences = self.pick_experiences(num_experiences)
        return (
            torch.from_numpy(np.array([
                self._state_producer(e.state_picture, e.state_vector)
                for e in experiences
            ], dtype=np.float32)).to(self.device),
            torch.from_numpy(np.array([e.action for e in experiences], dtype=np.float32)).to(self.device),
            torch.from_numpy(np.array([[e.reward] for e in experiences], dtype=np.float32)).to(self.device),
            torch.from_numpy(np.array([
                self._state_producer(e.next_state_picture, e.next_state_vector)
                for e in experiences
            ], dtype=np.float32)).to(self.device),
            torch.from_numpy(np.array([[e.done] for e in experiences], dtype=np.float32)).to(self.device),
        )

    def pick_experiences(self, num_experiences=None):
        if num_experiences is not None:
            batch_size = num_experiences
        else:
            batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)

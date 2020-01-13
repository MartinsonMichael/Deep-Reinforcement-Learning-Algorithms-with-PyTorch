from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class PictureProcessor(nn.Module):
    def __init__(self):
        super(PictureProcessor, self).__init__()

        self._conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=(8, 8),
            stride=(4, 4),
        )
        torch.nn.init.xavier_uniform_(self._conv1.weight)
        torch.nn.init.constant_(self._conv1.bias, 0)

        self._conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(4, 4),
            stride=(2, 2),
        )
        torch.nn.init.xavier_uniform_(self._conv2.weight)
        torch.nn.init.constant_(self._conv2.bias, 0)

        self._conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
        )
        torch.nn.init.xavier_uniform_(self._conv3.weight)
        torch.nn.init.constant_(self._conv3.bias, 0)

    def forward(self, state):
        x = F.relu(self._conv1(state))
        x = F.relu(self._conv2(x))
        x = F.relu(self._conv3(x))
        return x.view(x.size(0), -1)

    def get_out_shape_for_in(self, input_shape):
        return self.forward(torch.Tensor(np.zeros((1, *input_shape)))).shape[1]


class StateLayer(nn.Module):
    def __init__(self, state_description: Dict[str, Any], hidden_size, device):
        super(StateLayer, self).__init__()
        self._device = device

        self._picture_layer = None
        self._vector_layer = None

        self._state_layer_out_size = 0
        if 'picture' in state_description.keys() and state_description['picture'] is not None:
            self._picture_layer = PictureProcessor()
            self._state_layer_out_size += self._dense_s.get_out_shape_for_in(
                state_description['picture']
            )

        if 'vector' in state_description.keys() and state_description['vector'] is not None:
            self._vector_layer = nn.Linear(in_features=state_description['vector'], out_features=hidden_size)
            torch.nn.init.xavier_uniform_(self._vector_layer.weight)
            torch.nn.init.constant_(self._vector_layer.bias, 0)
            self._state_layer_out_size += hidden_size

    def get_out_shape_for_in(self):
        return self._state_layer_out_sizes

    def forward(self, state):
        state_pic = None
        if self._picture_layer is not None:
            state_pic = self._picture_layer(state['picture'] / 256)

        state_vec = None
        if self._vector_layer is not None:
            state_vec = self._vector_layer(state['vector'])

        if state_vec is not None and state_pic is not None:
            return torch.cat((state_pic, state_vec), dim=1)

        if state_pic is not None:
            return state_pic

        if state_vec is not None:
            return state_vec

        raise ValueError("state should be Dict['picture' : tensor or None, 'vector' : tensor or None]")


class QNet(nn.Module):
    def __init__(self, state_description: Dict[str, Any], action_size, hidden_size, device):
        super(QNet, self).__init__()
        self._device = device

        self._state_layer = StateLayer(state_description, hidden_size, device)

        self._dense_a = nn.Linear(in_features=action_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self._dense_a.weight)
        torch.nn.init.constant_(self._dense_a.bias, 0)

        self._dense2 = nn.Linear(
            in_features=hidden_size + self._state_layer.get_out_shape_for_in,
            out_features=hidden_size,
        )
        torch.nn.init.xavier_uniform_(self._dense2.weight)
        torch.nn.init.constant_(self._dense2.bias, 0)

        self._head1 = nn.Linear(in_features=hidden_size, out_features=action_size)
        torch.nn.init.xavier_uniform_(self._head1.weight)
        torch.nn.init.constant_(self._head1.bias, 0)

    def forward(self, state, action):
        s = F.relu(self._state_layer(state))
        a = F.relu(self._dense_a(action))
        x = torch.cat((s, a), 1)
        x = F.relu(self._dense2(x))
        x = self._head1(x)
        return x


class Policy(nn.Module):
    def __init__(self, state_description: Dict[str, Any], action_size, hidden_size, device):
        super(Policy, self).__init__()
        self._device = device
        self._action_size = action_size

        self._state_layer = StateLayer(state_description, hidden_size, device)

        self._dense2 = nn.Linear(in_features=self._state_layer.get_out_shape_for_in, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self._dense2.weight)
        torch.nn.init.constant_(self._dense2.bias, 0)

        self._head = nn.Linear(in_features=hidden_size, out_features=2 * action_size)
        torch.nn.init.xavier_uniform_(self._head.weight)
        torch.nn.init.constant_(self._head.bias, 0)

    def forward(self, state):
        x = F.relu(self._state_layer(state))
        x = F.relu(self._dense2(x))
        x = self._head(x)
        return x

from typing import Dict, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from torch.distributions import Normal


class PictureProcessor(nn.Module):
    def __init__(self, in_channels=3):
        super(PictureProcessor, self).__init__()

        self._conv1 = nn.Conv2d(
            in_channels=in_channels,
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
        return self.forward(torch.from_numpy(np.zeros(
            shape=(1, *tuple(input_shape)),
            dtype=np.float32))
        ).shape[1]


# class StateLayer(nn.Module):
#     def __init__(self, state_description: Dict[str, Any], hidden_size, device):
#         super(StateLayer, self).__init__()
#         self._device = device
#
#         self._picture_layer = None
#         self._vector_layer = None
#
#         print(f'state layer : {state_description}')
#
#         self._state_layer_out_size = 0
#         if 'picture' in state_description.keys() and state_description['picture'] is not None:
#             self._picture_layer = PictureProcessor()
#             self._state_layer_out_size += self._picture_layer.get_out_shape_for_in(
#                 state_description['picture']
#             )
#
#         if 'vector' in state_description.keys() and state_description['vector'] is not None:
#             self._vector_layer = nn.Linear(in_features=state_description['vector'], out_features=hidden_size)
#             torch.nn.init.xavier_uniform_(self._vector_layer.weight)
#             torch.nn.init.constant_(self._vector_layer.bias, 0)
#             self._state_layer_out_size += hidden_size
#
#     def get_out_shape_for_in(self):
#         return self._state_layer_out_size
#
#     def forward(self, state: Dict[str, Union[np.array, torch.FloatTensor]]):
#
#         state_pic = None
#         if self._picture_layer is not None:
#             state_pic = self._picture_layer(state['picture'] / 256)
#
#         state_vec = None
#         if self._vector_layer is not None:
#             state_vec = self._vector_layer(state['vector'])
#
#         if state_vec is not None and state_pic is not None:
#             return torch.cat((state_pic, state_vec), dim=1)
#
#         if state_pic is not None:
#             return state_pic
#
#         if state_vec is not None:
#             return state_vec
#
#         raise ValueError("state should be Dict['picture' : tensor or None, 'vector' : tensor or None]")
#

class NewStateLayer(nn.Module):
    def __init__(self, state_description: Union[dict, spaces.Box], hidden_size, device):
        super(NewStateLayer, self).__init__()
        self._device = device

        self._state_layer_out_size = 0

        self._picture_layer = None
        self._vector_layer = None

        print(f'state layer : {state_description}')

        if isinstance(state_description, (spaces.Dict, dict)):
            if 'picture' in state_description.keys() and state_description['picture'] is not None:
                self._picture_layer = PictureProcessor()
                self._state_layer_out_size += self._picture_layer.get_out_shape_for_in(
                    state_description['picture']
                )

            if 'vector' in state_description.keys() and state_description['vector'] is not None:
                self._vector_layer = nn.Linear(in_features=state_description['vector'], out_features=hidden_size)
                torch.nn.init.xavier_uniform_(self._vector_layer.weight)
                torch.nn.init.constant_(self._vector_layer.bias, 0)
                self._state_layer_out_size += hidden_size

        if isinstance(state_description, spaces.Box):
            if len(state_description.shape) == 3:
                self._picture_layer = PictureProcessor(state_description.shape[0])
                self._state_layer_out_size = self._picture_layer.get_out_shape_for_in(
                    state_description.shape
                )
            if len(state_description.shape) == 1:
                self._vector_layer = nn.Linear(in_features=state_description.shape[0], out_features=hidden_size)
                torch.nn.init.xavier_uniform_(self._vector_layer.weight)
                torch.nn.init.constant_(self._vector_layer.bias, 0)
                self._state_layer_out_size += hidden_size

    def get_out_shape_for_in(self):
        return self._state_layer_out_size

    def forward_dict(self, state: Dict[str, torch.FloatTensor]):
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

    def forward_picture(self, state: torch.FloatTensor):
        return self._picture_layer(state / 256)

    def forward_vector(self, state: torch.FloatTensor):
        return self._vector_layer(state)

    def _make_it_torch_tensor(self, x):
        if isinstance(x, (torch.FloatTensor, torch.Tensor, torch.DoubleTensor)):
            if len(x.shape) == 2 or len(x.shape) == 4:
                return x
            else:
                return x.unsqueeze_(0)
        if isinstance(x, np.ndarray):
            if len(x.shape) == 2 or len(x.shape) == 4:
                return torch.from_numpy(x.astype(np.float32))
            else:
                return torch.from_numpy(np.array([x]).astype(np.float32))

        print('state trouble')
        print(f'state type: {type(x)}')
        print(x)

        raise ValueError('add dict!')
        # if isinstance(x, dict):
        #     return {
        #         key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
        #         for key, value in x.items()
        #     }

    def forward(self, state: Union[Dict[str, torch.FloatTensor], torch.FloatTensor]):
        state = self._make_it_torch_tensor(state)

        if isinstance(state, dict):
            return self.forward_dict(state)

        if isinstance(state, (torch.FloatTensor, torch.Tensor)):
            if len(state.shape) == 4:
                return self.forward_picture(state)
            if len(state.shape) == 2:
                return self.forward_vector(state)

        print('state')
        print(f'state type : {type(state)}')
        print(f'state shape : {state.shape}')
        print(state)

        raise ValueError()


class QNet(nn.Module):
    def __init__(self, state_description: Dict[str, Any], action_size, hidden_size, device):
        super(QNet, self).__init__()
        self._device = device

        self._state_layer = NewStateLayer(state_description, hidden_size, device)

        self._dense_a = nn.Linear(in_features=action_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self._dense_a.weight)
        torch.nn.init.constant_(self._dense_a.bias, 0)

        self._dense2 = nn.Linear(
            in_features=hidden_size + self._state_layer.get_out_shape_for_in(),
            out_features=hidden_size,
        )
        torch.nn.init.xavier_uniform_(self._dense2.weight)
        torch.nn.init.constant_(self._dense2.bias, 0)

        self._head1 = nn.Linear(in_features=hidden_size, out_features=1)
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

        self._state_layer = NewStateLayer(state_description, hidden_size, device)

        self._dense2 = nn.Linear(in_features=self._state_layer.get_out_shape_for_in(), out_features=hidden_size)
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

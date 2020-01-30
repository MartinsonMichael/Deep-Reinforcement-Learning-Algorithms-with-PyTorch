from typing import Tuple, Union

import numpy as np


def heuristic_state_classifier(state: np.ndarray) -> str:
    if len(state.shape) == 3:
        if state.shape[0] == 3 or state.shape[2] == 3:
            return 'image'
        else:
            return 'both'
    if len(state.shape) == 1:
        return 'vector'

    # if len(state.shape) == 2 or len(state.shape) == 4:
    #     # hm, may be it is batch?
    #     return heuristic_state_classifier(state[0])

    raise ValueError(f'unknown state type : shape : {state.shape}')


def from_combined_state_to_image_vector(state: np.ndarray) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
    """
    Transform single state np.ndarray to
    0 - image np.ndarray with dtype np.uint8 and
    1 - vector np.ndarray with type np.float32
    """
    state_type = heuristic_state_classifier(state)
    if state_type == 'both':
        return _state_splitter__both(state)
    if state_type == 'vector':
        return None, state
    if state_type == 'image':
        return _prepare_image_to_buffer(state), None


def _state_splitter__both(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert len(state.shape) == 3, "state must have 3 dimensions"

    state_picture, state_vector_extended = np.split(state, [3], axis=0)
    state_vector = state_vector_extended[:, 0, 0]
    del state_vector_extended

    return _prepare_image_to_buffer(state_picture), state_vector


def from_image_vector_to_combined_state(image: Union[np.ndarray, None], vector: Union[np.ndarray, None]) -> np.ndarray:
    if image is not None and vector is not None:
        return np.concatenate([
                _prepare_image_to_model(image),
                np.ones(shape=list(image.shape[:-1]) + [len(vector)], dtype=np.float32) * vector
            ],
            axis=-1
        )
    if image is not None:
        return _prepare_image_to_model(image)
    if vector is not None:
        return vector

    raise ValueError('both image and vector are none')


def _prepare_image_to_buffer(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(a=image * 255, a_min=0.0, a_max=255.0).astype(np.uint8)


def _prepare_image_to_model(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255
    return image

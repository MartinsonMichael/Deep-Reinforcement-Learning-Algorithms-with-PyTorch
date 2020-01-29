from typing import Tuple

import numpy as np


def from_combined_state_to_image_vector(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert len(state.shape) == 3, "state must have 3 dimensions"

    state_picture, state_vector = np.split(state, [3], axis=2)

    return (state_picture * 256.0).astype(np.uint8), state_vector[0, 0, :]


def from_image_vector_to_combined_state(image: np.ndarray, vector: np.ndarray) -> np.ndarray:
    vector_channel = np.ones(shape=list(image.shape[:-1]) + [len(vector)], dtype=np.float32) * vector
    res = np.concatenate([image.astype(np.float32), vector_channel], axis=-1)
    return res

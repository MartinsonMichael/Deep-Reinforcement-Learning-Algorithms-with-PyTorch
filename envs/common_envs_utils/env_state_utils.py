from typing import Tuple

import numpy as np


def from_combined_state_to_image_vector(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert len(state.shape) == 3, "state must have 3 dimensions"

    state_picture, state_vector_extended = np.split(state, [3], axis=2)
    state_vector = state_vector_extended[:, 0, 0]
    del state_vector_extended

    return (state_picture * 256.0).astype(np.uint8), state_vector


def from_image_vector_to_combined_state(image: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return np.concatenate([
            image.astype(np.float32),
            np.ones(shape=list(image.shape[:-1]) + [len(vector)], dtype=np.float32) * vector
        ],
        axis=-1
    )

from typing import Optional

import chainerrl

from envs.common_envs_utils.env_wrappers import DiscreteWrapper
from envs.common_envs_utils.extended_env_wrappers import ExtendedMaxAndSkipEnv, FrameCompressor, \
    ImageWithVectorCombiner, ChannelSwapper, NumpyCaster, OnlyImageTaker, OnlyVectorTaker
from envs.gym_car_intersect_fixed import CarRacingHackatonContinuousFixed


def make_CarRacing_fixed_for_rainbow(settings_path: str, name: Optional[str] = None):
    def f():
        env = CarRacingHackatonContinuousFixed(settings_file_path=settings_path)
        # -> dict[(.., .., 3), (16)]
        env = chainerrl.wrappers.ContinuingTimeLimit(env, max_episode_steps=1200)
        env = ExtendedMaxAndSkipEnv(env, skip=4)
        env = FrameCompressor(env)
        # -> dict[(84, 84, 3), (16)]
        # env = OriginalStateKeeper(env, 'uncombined_state')
        env = ImageWithVectorCombiner(env)
        # -> Box(84, 84, 19)
        env = ChannelSwapper(env)
        # -> Box(19, 84, 84)
        env._max_episode_steps = 1200
        env = DiscreteWrapper(env)

        if name is not None:
            env.name = name

        return env

    return f


def make_CarRacing_fixed_combined_features(settings_path: str, name: Optional[str] = None):
    def f():
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
        env = NumpyCaster(env)
        env._max_episode_steps = 250

        if name is not None:
            env.name = name

        return env
    return f


def make_CarRacing_fixed_image_features(settings_path: str, name: Optional[str] = None):
    def f():
        env = CarRacingHackatonContinuousFixed(settings_file_path=settings_path)
        # -> dict[(.., .., 3), (16)]
        env = chainerrl.wrappers.ContinuingTimeLimit(env, max_episode_steps=250)
        env = ExtendedMaxAndSkipEnv(env, skip=4)
        env = FrameCompressor(env)
        # -> dict[(84, 84, 3), (16)]
        env = OnlyImageTaker(env)
        env = ChannelSwapper(env)
        # -> Box(19, 84, 84)
        env = NumpyCaster(env)
        env._max_episode_steps = 250

        if name is not None:
            env.name = name

        return env
    return f


def make_CarRacing_fixed_vector_features(settings_path: str, name: Optional[str] = None):
    def f():
        env = CarRacingHackatonContinuousFixed(settings_file_path=settings_path)
        # -> dict[(.., .., 3), (16)]
        env = chainerrl.wrappers.ContinuingTimeLimit(env, max_episode_steps=250)
        env = ExtendedMaxAndSkipEnv(env, skip=4)
        env = FrameCompressor(env)
        # -> dict[(84, 84, 3), (16)]
        env = OnlyVectorTaker(env)
        env = ChannelSwapper(env)
        # -> Box(19, 84, 84)
        env = NumpyCaster(env)
        env._max_episode_steps = 250

        if name is not None:
            env.name = name

        return env
    return f

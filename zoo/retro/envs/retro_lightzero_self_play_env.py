from zoo.atari.envs.atari_lightzero_env import AtariEnvLightZero
import copy
import gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from typing import List

##############################
## Work in progress
##############################

@ENV_REGISTRY.register('retro_lightzero_selfplay')
class RetroSelfPlayEnvLightZero(BaseEnv):
    config = AtariEnvLightZero.config

    @classmethod
    def default_config(cls: type) -> EasyDict:
        """
        Overview:
            Return the default configuration for the Atari LightZero environment.
        Arguments:
            - cls (:obj:`type`): The class AtariEnvLightZero.
        Returns:
            - cfg (:obj:`EasyDict`): The default configuration dictionary.
        """
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg):
        self._env = AtariEnvLightZero(cfg)

        self.current_player = 1

    def reset(self):
        obs = self._env.reset()
        self.current_player = 1

        obs['to_play'] = 1

        return obs

    def step(self, action):
        next_obs, reward, done, info = self._env.step(action)

        self.current_player = 3 - self.current_player  # Switch between players 1 and 2
        reward = reward if self.current_player == 2 else -reward
        next_obs['to_play'] = self.current_player

        return BaseEnvTimestep(next_obs, reward, done, info)

    @property
    def observation_space(self):
        return self._env.observation_space

    def observe(self) -> dict:
        obs = self._env.observe()

        obs['to_play'] = self.current_player

        return obs

    @property
    def legal_actions(self):
        return self._env.legal_actions

    def close(self) -> None:
        return self._env.close()

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        return self._env.seed(seed, dynamic_seed)

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._env.action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._env.reward_space

    def __repr__(self) -> str:
        return "LightZero SelfPlay Atari Env({})".format(self.cfg.env_id)

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.max_episode_steps = cfg.collect_max_episode_steps
        cfg.episode_life = True
        cfg.clip_rewards = True
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.max_episode_steps = cfg.eval_max_episode_steps
        cfg.episode_life = False
        cfg.clip_rewards = False
        return [cfg for _ in range(evaluator_env_num)]

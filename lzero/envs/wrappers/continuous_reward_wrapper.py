import gym

class ContinuousRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, reward: float = 0.01, max_reward: float = 0.25):
        """
        Arguments:
            - env (:obj:`gym.Env`): The environment to wrap.
        """
        super().__init__(env)
        self.reward = reward
        self.max_reward = max_reward
        self.current_enc_reward = 0.

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        if reward != 0.:
            if reward > 0.:
                reward -= self.current_enc_reward * 2 # The agent must try to get real reward as soon as possible
                if reward < 0.:
                    reward = self.reward * 2

            self.current_enc_reward = 0.
        elif self.current_enc_reward < self.max_reward:
            self.current_enc_reward += self.reward
            reward += self.reward
        else:
            reward -= self.reward

        return observation, reward, done, info

    def reset(self, **kwargs):
        self.current_enc_reward = 0.
        return self.env.reset(**kwargs)

def wrap_continuous_reward_wrapper_based_on_config(env, config):
    if config.continous_reward_wrapper and config.continous_reward_wrapper.enabled is True:
        env = ContinuousRewardWrapper(env, reward=config.continous_reward_wrapper.reward, max_reward=config.continous_reward_wrapper.max_reward)

    return env

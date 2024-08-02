import gym

class ContinuousRewardWrapper(gym.Wrapper):
    """
    A wrapper that modifies the reward structure of an environment to provide continuous feedback.

    This wrapper is useful in environments with sparse rewards, where the agent might struggle
    to learn due to infrequent feedback. It adds small incremental rewards over time to encourage
    exploration and maintain engagement, while still preserving the importance of achieving the
    main objectives.

    Use this wrapper when:
    1. The environment has sparse rewards
    2. You want to encourage more consistent exploration
    3. Learning is slow due to infrequent feedback
    4. You want to balance immediate feedback with long-term goals
    """

    def __init__(self, env: gym.Env, reward: float = 0.01, max_reward: float = 0.25):
        """
        Args:
            env (gym.Env): The environment to wrap.
            reward (float): The small reward to add each step (default: 0.01).
            max_reward (float): The maximum cumulative additional reward (default: 0.25).
        """
        super().__init__(env)
        self.reward = reward
        self.max_reward = max_reward
        self.current_enc_reward = 0.

    def step(self, action):
        """
        This method adds small rewards over time, but ensures that achieving the
        main objectives (i.e., getting a non-zero reward from the base environment)
        remains the primary goal.
        """

        observation, reward, done, info = self.env.step(action)

        if reward != 0.:
            # If there's a real reward, subtract the accumulated small rewards
            # to maintain the relative importance of the main objective
            if reward > 0.:
                reward -= self.current_enc_reward * 2 # Encourage getting real rewards quickly
                if reward < 0.:
                    reward = self.reward * 2 # Ensure some positive feedback for achieving goals

            self.current_enc_reward = 0.
        elif self.current_enc_reward < self.max_reward:
            # Add small reward if no real reward and below the maximum
            self.current_enc_reward += self.reward
            reward += self.reward
        else:
            # Discourage staying in the same state too long
            reward -= self.reward

        return observation, reward, done, info

    def reset(self, **kwargs):
        self.current_enc_reward = 0.
        return self.env.reset(**kwargs)

def wrap_continuous_reward_wrapper_based_on_config(env: gym.Env, config):
    """
   Apply the ContinuousRewardWrapper to an environment based on configuration.

   This function is useful for easily enabling/disabling the wrapper and setting
   its parameters through a configuration object, allowing for more flexible
   experimentation without changing the main code.

   Args:
       env (gym.Env): The environment to potentially wrap.
       config: A configuration object containing wrapper settings.

   Returns:
       gym.Env: The wrapped environment if enabled in config, otherwise the original environment.
   """

    if config.continous_reward_wrapper and config.continous_reward_wrapper.enabled is True:
        env = ContinuousRewardWrapper(env, reward=config.continous_reward_wrapper.reward, max_reward=config.continous_reward_wrapper.max_reward)

    return env

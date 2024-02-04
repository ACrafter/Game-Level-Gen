import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN

import seaborn as sns
import matplotlib.pyplot as plt
import sympy
from time import time as gettime


class Generator(gym.Env):
    """
        The Environment represents the Generator,
        Goal: Generate levels that responds to the player's experience (Harder for better players, Easier for bad ones),
    """

    def __init__(self, size=5, player=None, player_t=200_000, generator_t=100_000, player_max_t=1_000_000):
        """
        Initializes base parameters, the rest are initialized in the Reset()
        :param size: Board size
        :param player: Player Environment Class
        :param player_t: Number of training steps for the player model in a single eps
        :param generator_t: Number of training steps for the generator in a single eps
        :param player_max_t: Max number of training steps for the player
        :param player_lr: The learning rate of the player model
        """

        self.size = size
        self.player = player
        self.player_steps = player_t
        self.generator_steps_per_eps = generator_t
        self.player_max_steps = player_max_t
        self.player_total_trained_steps = 0

        self.observation_space = spaces.Dict({
            "lost_lives": spaces.Box(0, 1, shape=(1,), dtype=int),
            "time_spent": spaces.Box(0, 60, shape=(1,), dtype=int),
            "current_max": spaces.Box(0, 100, shape=(1,), dtype=int),
            "prime_range": spaces.Box(0, 100, shape=(1,), dtype=int),
            "last_level_passed": spaces.Discrete(2)
        })

        # Increase Max Range, Increase Prime Range, Hold, Decrease Max Range, Decrease Prime Range
        self.action_space = spaces.Discrete(8)  # Can Be Increased To provide more control

        self.reset()

    def _get_obs(self):
        return {
            "lost_lives": self.lost_lives,
            "time_spent": self.time_spent,
            "current_max": self.max,
            "prime_range": self.prime_range,
            "last_level_passed": self.failed
        }

    def reset(self, seed=None, option=None):
        """
        Initializes other parameters for the class & handles training and pre-training of the agent
        :param seed: ---
        :param option: ---
        :return: Current Observation & Into
        """
        self.min = 1
        self.max = 5
        self.prime_range = 5
        self.prime_percent = 0.3
        self.lost_lives = None
        self.time_spent = None
        self.failed = None

        self.generator_eps_steps = 0
        self.player_trained_steps = 0

        if self.player_total_trained_steps < self.player_max_steps:

            print(f"Started Player Training, Steps: {self.player_steps}")
            start_time = gettime()
            print(f"Trained for: {self.player_trained_steps}")
            while self.player_trained_steps < self.player_steps:
                self.train_player()
            end_time = gettime()
            print(f"Player Training Ended, Time Taken: {int(end_time - start_time) // 60} mins")
            self.player_total_trained_steps += self.player_steps

        else:
            print(f"============== Player Reached Max Training Steps: {self.player_max_steps}")

        observation = self._get_obs()
        info = {}

        return observation, info

    def train_player(self):
        self.player.reset()
        self.player_model = DQN('MultiInputPolicy', env=self.player, verbose=1, tensorboard_log='./player_log/')

        self.player_model.learn(total_timesteps=1)
        self.player_trained_steps += self.player_steps

    def step(self, action):
        self.generator_eps_steps += 1
        total_reward = 0

        if action == 0:
            self.max += 5
            if self.max > 100:
                total_reward -= 10
                self.max = 100

        elif action == 1:
            self.prime_range += 5
            if self.prime_range > 100:
                total_reward -= 10
                self.prime_range = 100

        elif action == 2:
            self.prime_percent += 0.1
            if self.prime_percent > 0.8:
                total_reward -= 10
                self.prime_percent = 0.8

        elif action == 4:
            self.max -= 5
            if self.max == 0:
                total_reward -= 10
                self.max = 5

        elif action == 5:
            self.prime_range -= 5
            if self.prime_range == 0:
                total_reward -= 10
                self.prime_range = 5

        elif action == 6:
            self.prime_percent -= 0.1
            if self.prime_percent == 0.2:
                total_reward -= 10
                self.prime_percent = 0.2

        level, primes = self.gen_level()
        lives, time, failed = self.eval_level(level)

        self.lost_lives = 10 - lives
        self.time_spent = time

        if 1 < self.lost_lives < 5:
            total_reward += 50
        else:
            total_reward -= 20

        if 30 < self.time_spent < 60:
            total_reward += 50
        else:
            total_reward -= 20

        observation = self._get_obs()
        terminated = self.generator_eps_steps == self.generator_steps_per_eps

        return observation, total_reward, terminated, False, {}

    def gen_level(self):
        prime_numbers = list(sympy.primerange(1, self.prime_range + 1))

        possible_non_primes = np.arange(1, self.max)
        non_prime_numbers = possible_non_primes[~np.isin(possible_non_primes, prime_numbers)]

        number_of_primes = int(self.prime_percent * self.size * self.size)
        prime_cells = np.random.choice(prime_numbers, number_of_primes, replace=True)

        number_of_non_primes = (self.size * self.size) - number_of_primes
        non_prime_cells = np.random.choice(non_prime_numbers, number_of_non_primes, replace=True)

        board = np.append(prime_cells, non_prime_cells)
        np.random.shuffle(board)
        board = np.reshape(board, (self.size, self.size))

        return board

    def eval_level(self, level):
        print('=============== Player Evaluating ===============')
        env = self.player(preset=level)
        state, info = env.reset()
        done = False
        start_time = gettime()
        t = 0

        while not done and t < 60:
            t = gettime() - start_time
            action, _state = self.player_model.predict(state)
            state, reward, done, clipped, info = env.step(action)

        end_time = gettime()
        remaining_lives = env.remaining_lives
        time = int(end_time - start_time)
        failed = False

        if remaining_lives == 0 or time == 60:
            failed = True

        print(f'=============== Player Evaluating Done, Level Failed: {failed} ===============')
        return remaining_lives, time, failed

    def render(self):
        print(
            f"Current Gen Step: {self.generator_eps_steps}, Current Diff: {self.max}, Latest Remaining Lives: "
            f"{self.lost_lives}, Latest Remaining "
            f"Time: {self.time_spent}"
        )


G = Generator()
for i in range(10):
    print(G.gen_level())

import os
import random

import numpy as np
import pygame

import gymnasium as gym
import sympy
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class BetterEnv(gym.Env):
    def __init__(self, size=5):
        self.size = size
        self.prime_numbers = np.array(
            [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
             89, 97])
        self.window = None
        self.clock = None
        self.board = ([], [])
        self.score = 0
        self.lives = 10
        self.level = 1
        self.player_pos = ([], [])
        self.current_primes = []
        self.active_primes = []  # Same as current but only for the environment
        self.current_number = 0
        self.steps_since_last_eat = 0
        self.remaining_number_of_primes = 0

        # Render Variables
        self.steps = 0
        self.eaten_numbers = []

        # Observation Space
        self.observation_space = spaces.Dict({
            "board": spaces.Box(0, 100, shape=(self.size, self.size), dtype=int),
            "lives": spaces.Box(0, 10, shape=(1,), dtype=int),
            "player_pos": spaces.Box(0, self.size, shape=(2,),
                                     dtype=int),
            "current_number": spaces.Discrete(100),
            "remaining_number_of_primes": spaces.Discrete((self.size * self.size))
        })

        # Action Space
        self.action_space = spaces.Discrete(5)

        pygame.init()
        pygame.display.init()
        pygame.font.init()
        self.cell_size = 75
        self.screen_size = self.size * self.cell_size
        self.screen = None

    # The Reset, Level Data & Board Functions -- Environment

    # Gets the Params for the new level
    def new_level_data(self):
        min_n = 1
        max_n = self.level + 1

        min_p_index = 0
        max_p_index = self.level

        num_p = int((self.size * self.size) * 0.3)

        data = [min_n, max_n, min_p_index, max_p_index, num_p]
        return data

    def _get_obs(self):
        return {
            "board": self.board,
            "player_pos": self.player_pos,
            "lives": self.lives,
            "current_number": self.current_number,
            "remaining_number_of_primes": self.remaining_number_of_primes
        }

    # Create the new level board
    def create_board(self):
        min_n, max_n, min_p, max_p, p_count = self.new_level_data()
        possible_non_primes = np.arange(min_n, max_n)
        non_p = np.random.choice(possible_non_primes[~np.isin(possible_non_primes, self.prime_numbers)]
                                 , size=(self.size * self.size) - p_count, replace=True)
        possible_primes = self.prime_numbers[min_p: max_p + 1]
        p = np.random.choice(possible_primes, size=p_count, replace=True)
        self.active_primes = p

        board = np.append(non_p, p)
        np.random.shuffle(board)
        board = np.reshape(board, (self.size, self.size))
        return board

    def reset(self, seed=None, options=None):
        print("===============\n"
              "Game Got Reset\n"
              "===============")
        self.board = self.create_board()
        self.player_pos = np.array([0, 0])
        self.lives = 10
        self.current_number = self.board[self.player_pos[0]][self.player_pos[1]]
        self.steps_since_last_eat = 0
        self.remaining_number_of_primes = int((self.size * self.size) * 0.3)

        observation = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, info

    # The Step, Reward & Termination Functions -- Agent Actions

    def step(self, action):
        reward = 0
        if action == 0:  # Up
            if self.player_pos[1] == 0:
                reward = -12 + self.steps_since_last_eat
            else:
                self.player_pos[1] -= 1
                reward = -10 + self.steps_since_last_eat
        if action == 1:  # Down
            if self.player_pos[1] == self.size - 1:
                reward = -12 + self.steps_since_last_eat
            else:
                self.player_pos[1] += 1
                reward = -10 + self.steps_since_last_eat
        if action == 2:  # Left
            if self.player_pos[0] == 0:
                reward = -12 + self.steps_since_last_eat
            else:
                self.player_pos[0] -= 1
                reward = -10 + self.steps_since_last_eat
        if action == 3:  # Right
            if self.player_pos[0] == self.size - 1:
                reward = -12 + self.steps_since_last_eat
            else:
                self.player_pos[0] += 1
                reward = -10 + self.steps_since_last_eat

        if action == 4:  # Eating
            self.eaten_numbers.append({self.current_number, sympy.isprime(self.current_number)})

            if self.current_number == 0:
                reward = -100
            else:
                self.board[self.player_pos[0]][self.player_pos[1]] = 0
                if self.current_number in self.prime_numbers:
                    self.steps_since_last_eat = 0

                    # V6
                    self.remaining_number_of_primes -= 1

                    self.score += 1
                    reward = 50
                else:
                    self.lives -= 1
                    reward = -20
        else:
            self.steps_since_last_eat -= 0.1

        self.current_number = self.board[self.player_pos[0]][self.player_pos[1]]
        terminated = self.lives == 0 or self.remaining_number_of_primes == 0  # V6
        observation = self._get_obs()

        if self.remaining_number_of_primes == 0:  # V6
            print(f"Remaining Primes: {self.remaining_number_of_primes}\nCurrent Board: {self.board}")
            print("===============\n"
                  "Level Up\n"
                  "===============")
            if self.level != 15:
                self.level += 1

            reward = 40

        if self.lives == 0:
            print("===============\n"
                  "Game Over\n"
                  "===============")
            reward = -20

        if self.steps == 250:
            self.render()

        self.steps += 1

        return observation, reward, terminated, False, {}

    def render(self):
        print(self.eaten_numbers)
        self.steps = 0
        self.eaten_numbers = []


Env = BetterEnv()
Env.reset()
for i in range(100):
    Env.step(Env.action_space.sample())

import numpy as np
import gymnasium as gym
import pygame.time
import sympy
from gymnasium import spaces
from Environments.Player.helper import gen_board

ACTION_TO_TEXT = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'EAT'}


class Player(gym.Env):
    def __init__(self, size=5, max_num=1000, preset_level=None, training=True):
        self.size = size
        self.max_num = max_num
        self.preset_level = preset_level
        self.training = training

        self.max_lives = 10
        self.current_max_num = 5
        self.max_time = 30
        self.level = 1

        self.ideal_prime_count = int(self.size * self.size * 0.3)
        self.accepted_prime_variation = int(self.size * self.size * 0.1)
        self.prime_count = np.array([self.ideal_prime_count - self.accepted_prime_variation, self.ideal_prime_count,
                                     self.ideal_prime_count + self.accepted_prime_variation])
        self.remaining_number_of_primes = 0

        self.steps_since_last_eat = 0
        self.lives_lost = 0
        self.time_spent = 0

        self.is_prime = None
        self.start_time = None
        self.board = None
        self.player_pos = None
        self.current_number = None

        # Render Variables
        self.steps = 0
        self.eaten_numbers = []
        self.actions_taken = []

        # Gen Variables
        self.total_levels_played = -1
        self.total_lives_lost = 0
        self.total_time_spend = 0
        self.max_num_appeared = 0
        self.number_of_primes = 0

        # Observation Space
        self.observation_space = spaces.Dict({
            "board": spaces.Box(0, self.max_num, shape=(self.size, self.size), dtype=int),
            "lives_lost": spaces.Box(0, self.max_lives, shape=(1,), dtype=int),
            "player_pos": spaces.Box(0, self.size, shape=(2,),
                                     dtype=int),
            "current_number": spaces.Box(0, self.max_num, shape=(1,), dtype=int),
            "is_prime": spaces.Box(0, 1, shape=(1, ), dtype=int)
        })

        # Action Space
        self.action_space = spaces.Discrete(5)

    def _get_obs(self):
        return {
            "board": np.array([self.board]),
            "player_pos": np.array([self.player_pos]),
            "lives_lost": np.array([self.lives_lost]),
            "current_number": np.array([self.current_number]),
            "is_prime": np.array([self.is_prime])
        }

    def reset(self, seed=None, options=None):
        # Gen Info
        self.total_levels_played += 1
        self.total_time_spend += self.time_spent
        self.number_of_primes = self.remaining_number_of_primes
        self.total_lives_lost += self.lives_lost

        # Level Info
        if self.preset_level is None:
            self.remaining_number_of_primes = np.random.choice(self.prime_count)
            self.board, _ = gen_board(self.size, self.current_max_num, self.remaining_number_of_primes)
        else:
            self.board, _, self.remaining_number_of_primes = self.preset_level

        # Agent Info
        self.lives_lost = 0
        self.time_spent = 0
        self.steps_since_last_eat = 0
        self.player_pos = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
        self.current_number = self.board[self.player_pos[0]][self.player_pos[1]]
        self.is_prime = 1 if sympy.isprime(self.current_number) else 0
        self.eaten_numbers = []
        self.actions_taken = []

        # Render Info
        self.steps += 1

        obs = self._get_obs()
        self.start_time = pygame.time.get_ticks() // 1000
        return obs, {}

    def step(self, action):
        reward = 0
        if action == 0:  # Up
            self.player_pos[0] = max(self.player_pos[0] - 1, 0)

        if action == 1:  # Down
            self.player_pos[0] = min(self.player_pos[0] + 1, 4)

        if action == 2:  # Left
            self.player_pos[1] = max(self.player_pos[1] - 1, 0)

        if action == 3:  # Right
            self.player_pos[1] = min(self.player_pos[1] + 1, 4)

        if action == 4:  # Eating
            self.eaten_numbers.append(
                    [self.current_number, True if sympy.isprime(self.current_number) else False])

            if self.is_prime == 1:
                reward += 30
                self.remaining_number_of_primes -= 1
            else:
                reward -= 10
                self.lives_lost += 1

            self.board[self.player_pos[0]][self.player_pos[1]] = 0

        current_time = pygame.time.get_ticks() // 1000
        self.time_spent = min(current_time - self.start_time, 30)

        if self.lives_lost == 10:
            self.current_max_num = max(self.current_max_num - 5, 5)

        if self.remaining_number_of_primes == 0:
            self.current_max_num = min(self.current_max_num + 5, 1000)

        if self.training:
            self.actions_taken.append([ACTION_TO_TEXT[action], f'Current Number: {self.current_number}'])

        self.max_num_appeared = max(self.max_num_appeared, self.current_max_num)
        self.current_number = self.board[self.player_pos[0]][self.player_pos[1]]
        self.is_prime = 1 if sympy.isprime(self.current_number) else 0
        done = self.lives_lost == 10 or self.time_spent == 30 or self.remaining_number_of_primes == 0
        obs = self._get_obs()

        if done and self.steps == 20 and self.training:
            self.render()

        return obs, reward, done, False, {}

    def render(self):
        print('Current Board: ')
        for row in self.board:
            print(row)

        print(f"Agent's Actions: {self.actions_taken}")
        print(f'Eaten Numbers: {self.eaten_numbers}')
        print(f'Lives Lost: {self.lives_lost}, Time Spent: {self.time_spent}, Current Max: {self.current_max_num}, '
              f'Remaining Primes: {self.remaining_number_of_primes}')

        self.steps = 0

    def get_average_lives_lost(self):
        if self.total_levels_played > 0:
            return self.total_lives_lost // self.total_levels_played
        else:
            return 0

    def get_average_time_lost(self):
        if self.total_levels_played > 0:
            return self.total_time_spend // self.total_levels_played
        else:
            return 0

    def get_max_number_appeared(self):
        return self.max_num_appeared

    def get_remaining_primes(self):
        return self.number_of_primes

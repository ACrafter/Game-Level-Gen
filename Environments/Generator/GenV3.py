import random
import time

import gymnasium as gym
import sympy
from gymnasium import spaces
import numpy as np
from Environments.Generator.helper import gen_level, calculate_rewards_PCGRL, calculate_rewards_with_in_range_method, \
    check_primes_and_max


# TODO: Add the following to the observation space: max_prime_count, min_prime_count, current_prime_number. Also add
#  mins and maxs for the rest of our parameters. Add a max_of_same, to represent the max amount of the same number in
#  a row

class Generator(gym.Env):
    """
        Generate levels based on the agent performance
        Steps:
            1) Train the agent for a given amount of steps on randomly generated levels and collect data
                1.1) Average time to complete a level, average lost lives, etc..
                1.2) The randomly generated levels are generated with sequential difficulty
                    If the agent passes a level the max is increased if it loses the max is reset

            2) Once done freeze the agent, and train the generator with the goal of creating more suitable levels
                2.1) The generator receives a randomly generated level and starts taking actions
                2.2) Each step the generator takes is evaluated by the agent,
                    we run the agent on the modified level, compare the time and lives lost with the averages
                2.3) Based on the evaluation of the agent and the averages the reward is calculated
                    If the agent had low averages the model needs to make the level easier
                    If the agent had higher averages the model needs to make the level harder
                2.4) The Generator is allowed a set number of changes (% Change) each episode once done it resets
                2.5) Resetting means generating the SAME LEVEL by passing the SAME SEED.

            3) Once the Generator has trained we freeze it and use it to generate levels for the agent to train on.
            4) Once the agent has trained we repeat the process again.
    """

    def __init__(self, player, steps_before_freeze, size=5, max_num=100):
        """
        Initializes base parameters, the rest are initialized in the Reset()
        :param size: Board size
        """

        super(Generator, self).__init__()
        self.size = size
        self.max_num = max_num
        self.player = player
        self.max_iterations = steps_before_freeze
        self.current_iteration = 0

        # Hidden details for generation
        self._seed = None
        self._level = None
        self._past_seeds = []
        self._past_levels = np.array([])

        self._playable = 1
        self._ideal_lives_lost = 0
        self._ideal_time_spent = 0
        self._ideal_max_number = 0
        self._ideal_prime_count = int(self.size * self.size * 0.3)
        self._accepted_lives_variation = 1
        self._accepted_time_variation = 5
        self._accepted_max_variation = 5
        self._accepted_prime_count_variations = int(self.size * self.size * 0.1)

        self._changes_made = 0
        self._steps_taken = 0
        self._max_changes = int((self.size * self.size) - 1)
        self._max_steps = self._max_changes * self.size * self.size
        self._change_map = np.zeros((self.size, self.size))

        self._freeze = True

        # Agent related details
        self.current_number = 0
        self.current_lost_lives = 0
        self.current_time = 0
        self.current_prime_count = 0
        self.old_number = 0
        self.old_lives = 0
        self.old_time = 0

        self.observation_space = spaces.Dict({
            "map": spaces.Box(low=-1, high=self.max_num, shape=(self.size, self.size), dtype=int),
            "max_max": spaces.Box(low=0, high=self.max_num, dtype=int, shape=(1,)),
            "max_time": spaces.Box(low=0, high=30, dtype=float, shape=(1,)),
            "max_lives": spaces.Box(low=0, high=10, dtype=float, shape=(1,)),
            "max_prime_count": spaces.Box(low=5, high=15, dtype=int, shape=(1,)),
            "min_max": spaces.Box(low=0, high=self.max_num, dtype=int, shape=(1,)),
            "min_time": spaces.Box(low=0, high=30, dtype=float, shape=(1,)),
            "min_lives": spaces.Box(low=0, high=10, dtype=float, shape=(1,)),
            "min_prime_count": spaces.Box(low=5, high=15, dtype=int, shape=(1,)),
            "current_max": spaces.Box(low=0, high=self.max_num, dtype=int, shape=(1,)),
            "current_lives": spaces.Box(low=0, high=10, dtype=float, shape=(1,)),
            "current_time": spaces.Box(low=0, high=30, dtype=float, shape=(1,)),
            "current_prime_count": spaces.Box(low=5, high=15, dtype=int, shape=(1,)),
            "old_lives": spaces.Box(low=0, high=10, dtype=float, shape=(1,)),
            "old_time": spaces.Box(low=0, high=30, dtype=float, shape=(1,)),
            "playable_level": spaces.Box(low=0, high=2, dtype=int, shape=(1,)),
            "prime_mask": spaces.Box(low=-1, high=self.max_num, shape=(self.size, self.size), dtype=int)
        })

        # Using Narrow Representation
        self.action_space = spaces.Discrete(100)

    def get_observation(self):
        return {
            "map": self._level.copy(),
            "max_max": self._ideal_max_number + self._accepted_max_variation,
            "max_time": self._ideal_time_spent + self._accepted_time_variation,
            "max_lives": self._ideal_lives_lost + self._accepted_lives_variation,
            "max_prime_count": self._ideal_prime_count + self._accepted_prime_count_variations,
            "min_max": self._ideal_max_number - self._accepted_max_variation,
            "min_time": self._ideal_time_spent - self._accepted_time_variation,
            "min_lives": self._ideal_lives_lost - self._accepted_lives_variation,
            "min_prime_count": self._ideal_prime_count - self._accepted_prime_count_variations,
            "current_max": self.current_number,
            "current_lives": self.current_lost_lives,
            "current_time": self.current_time,
            "current_prime_count": self.current_prime_count,
            "old_lives": self.old_lives,
            "old_time": self.old_time,
            "playable_level": self._playable,
            "prime_mask": np.array([[int(sympy.isprime(num)) for num in row] for row in self._level], dtype=int)
        }

    def reset(self, seed=None, options=None):
        if self.current_iteration >= self.max_iterations:
            self._freeze = True

        if self._freeze:
            print("Player Training Started!")
            self._ideal_lives_lost, self._ideal_time_spent, self._ideal_max_number = self.player.train()
            self._freeze = False
            self._seed = [random.randint(0, 2 ** 32 - 1) for _ in range(1000)]
            self._past_seeds.append(self._seed[random.randint(0, 1000)])
            print(f"Player Training Ended. Stats: Lives Lost {self._ideal_lives_lost}, "
                  f"Time Cost {self._ideal_time_spent}, "
                  f"Max To Appear {self._ideal_max_number}.")
            self.current_iteration = 0

        self._level = gen_level(self.size, self._seed)
        self._changes_made = 0
        self._steps_taken = 0
        self.current_prime_count = 0
        self._change_map = np.zeros((self.size, self.size))

        observation = self.get_observation()
        return observation, {}

    def step(self, action):
        rewards = - 40
        lvl_board = None
        eaten_numbers = None
        change, new_value = self.update(action, self._changes_made)
        if change and self._changes_made < self._max_changes:
            old_value = self.current_number
            self._level[self._changes_made // self.size][self._changes_made % self.size] = new_value
            self.current_number = new_value

            if sympy.isprime(self.current_number):
                self.current_prime_count += 1

            self._changes_made += 1
            rewards = calculate_rewards_with_in_range_method(self.current_number, self._ideal_max_number,
                                                             self._accepted_max_variation, 0.1)

            if not self.is_playable():
                rewards *= 0.75

        if change and self._changes_made == self._max_changes and self.is_playable():
            old_value = self.current_number
            self._level[self._changes_made // self.size][self._changes_made % self.size] = new_value
            self.current_number = new_value
            primes, max_number = check_primes_and_max(self._level)
            self.current_lost_lives, self.current_time, lvl_board, eaten_numbers = self.player.play(self._level.copy(),
                                                                                                    primes, max_number)
            rewards = calculate_rewards_PCGRL(self.current_lost_lives, self.old_lives, self._ideal_lives_lost,
                                              self._accepted_lives_variation) + \
                      calculate_rewards_PCGRL(self.current_time, self.old_time, self._ideal_time_spent,
                                              self._accepted_time_variation) + \
                      calculate_rewards_with_in_range_method(self.current_number, self._ideal_max_number,
                                                             self._accepted_max_variation, 0.1)

        observation = self.get_observation()
        done = self._changes_made >= self._max_changes or self._steps_taken >= self._max_steps

        self._steps_taken += 1
        self.current_iteration += 1

        if done:
            self.render(lvl_board, eaten_numbers)

        return observation, rewards, done, False, {}

    def update(self, action, cords):
        new_value = action + 1
        change = True if self._level[cords // self.size][cords % self.size] != action else False
        return change, new_value

    """
        Ensures that the level has the right prime count & has the right variety
    """

    def is_playable(self):
        if self._ideal_prime_count - self._accepted_prime_count_variations <= self.current_prime_count \
                <= self._ideal_prime_count + self._accepted_prime_count_variations:
            self._playable = 2
            return True

        # Uniqueness enforcement #1
        # for row in self._level:
        #     if len(set(row)) != len(row):
        #         self._playable = 0
        #         return False
        #
        #     # Check uniqueness in columns
        # for col in self._level.T:  # Transpose to iterate over columns
        #     if len(set(col)) != len(col):
        #         self._playable = 0
        #         return False

        self._playable = 0
        return False

    def render(self, lvl_board=None, eaten_numbers=None):
        if lvl_board is None:
            lvl_board = []

        if eaten_numbers is None:
            eaten_numbers = []

        print(f'Current Iteration: {self.current_iteration}')
        print(f'Base Level: ')
        for row in gen_level(self.size, self._seed):
            print(row)

        print(f'Final Level: ')
        for row in self._level:
            print(row)

        print('Played Level: ')
        for row in lvl_board:
            print(row)

        print('Eaten Numbers: ')
        print(eaten_numbers)

        print(f'Variations: (Lives, Time, Max)')
        for i in self._get_variations():
            print(i)

    def _get_variations(self):
        return [[self.current_lost_lives, self._ideal_lives_lost], [self.current_time, self._ideal_time_spent],
                [max(self._level.flatten()), self._ideal_max_number]]

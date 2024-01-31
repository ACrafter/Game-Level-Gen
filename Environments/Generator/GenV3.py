import random

import gymnasium as gym
import sympy
from gymnasium import spaces
import numpy as np
from Environments.Generator.helper import gen_level, calculate_rewards_with_in_range_method, \
    check_primes

EASY_LIVES, EASY_LIVES_VARIATION, EASY_MAX, EASY_MAX_VARIATION = 1, 1, 2, 10
MID_LIVES, MID_LIVES_VARIATION, MID_MAX, MID_MAX_VARIATION = 2, 2, 10, 10
HARD_LIVES, HARD_LIVES_VARIATION, HARD_MAX, HARD_MAX_VARIATION = 3, 3, 15, 10


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

    def __init__(self, training, player, steps_before_freeze, difficulty=None, size=5, max_num=100):
        """
        Initializes base parameters, the rest are initialized in the Reset()
        :param size: Board size
        """

        super(Generator, self).__init__()
        self.size = size
        self.is_training = training
        self.difficulty = difficulty
        self.max_num = max_num
        self.player = player
        self.max_iterations = steps_before_freeze
        self.current_iteration = 0

        # Hidden details for generation
        self._seed = None
        self._level = None
        self._past_seeds = []
        self._past_levels = np.array([])

        self._playable = 0
        self._ideal_lives_lost, self._ideal_max_number, self._accepted_lives_variation, self._accepted_max_variation = \
            self.set_difficulty()
        self._ideal_prime_count = int(self.size * self.size * 0.3)  # 7
        self._accepted_prime_count_variations = int(self.size * self.size * 0.1)  # 2

        self._changes_made = 0
        self._steps_taken = 0
        self._max_changes = int((self.size * self.size) - 1)
        self._max_steps = self._max_changes * self.size * self.size
        self._change_map = np.zeros((self.size, self.size))

        self._freeze = True

        # Agent related details
        self.remaining_primes = 0
        self.current_number = 0
        self.current_lost_lives = 0
        self.current_prime_count = 0

        self.observation_space = spaces.Dict({
            "map": spaces.Box(low=-1, high=self.max_num, shape=(self.size, self.size), dtype=int),
            "max_max": spaces.Box(low=0, high=self.max_num, dtype=int, shape=(1,)),
            "max_lives": spaces.Box(low=0, high=10, dtype=float, shape=(1,)),
            "max_prime_count": spaces.Box(low=5, high=15, dtype=int, shape=(1,)),
            "min_max": spaces.Box(low=0, high=self.max_num, dtype=int, shape=(1,)),
            "min_lives": spaces.Box(low=0, high=10, dtype=float, shape=(1,)),
            "min_prime_count": spaces.Box(low=5, high=15, dtype=int, shape=(1,)),
            "current_max": spaces.Box(low=0, high=self.max_num, dtype=int, shape=(1,)),
            "current_lives": spaces.Box(low=0, high=10, dtype=float, shape=(1,)),
            "current_prime_count": spaces.Box(low=5, high=15, dtype=int, shape=(1,)),
            "playable_level": spaces.Box(low=0, high=2, dtype=int, shape=(1,)),
            "prime_mask": spaces.Box(low=-1, high=self.max_num, shape=(self.size, self.size), dtype=int)
        })

        # Using Narrow Representation
        self.action_space = spaces.Discrete(100)

    def get_observation(self):
        return {
            "map": self._level.copy(),
            "max_max": [min(self._ideal_max_number + self._accepted_max_variation, 100)],
            "max_lives": [self._ideal_lives_lost + self._accepted_lives_variation],
            "max_prime_count": [self._ideal_prime_count + self._accepted_prime_count_variations],
            "min_max": [max(self._ideal_max_number - (self._accepted_max_variation * 2), 1)],
            "min_lives": [self._ideal_lives_lost - self._accepted_lives_variation],
            "min_prime_count": [self._ideal_prime_count - self._accepted_prime_count_variations],
            "current_max": [self.current_number],
            "current_lives": [self.current_lost_lives],
            "current_prime_count": [self.current_prime_count],
            "playable_level": [self._playable],
            "prime_mask": np.array([[int(sympy.isprime(num)) if
                                     self._ideal_max_number - self._accepted_max_variation <= num
                                     <= self._ideal_max_number + self._accepted_max_variation else 0
                                     for num in row] for row in self._level], dtype=int)
        }

    def set_difficulty(self):
        if self.difficulty is None:
            self.difficulty = random.randint(1, 3)

        if self.difficulty == 0:
            return EASY_LIVES, EASY_LIVES_VARIATION, EASY_MAX, EASY_MAX_VARIATION
        elif self.difficulty == 1:
            return MID_LIVES, MID_LIVES_VARIATION, MID_MAX, MID_MAX_VARIATION
        else:
            return HARD_LIVES, HARD_LIVES_VARIATION, HARD_MAX, HARD_MAX_VARIATION

    def reset(self, seed=None, options=None):
        if self.current_iteration >= self.max_iterations:
            self._freeze = True

        if self._freeze and self.is_training:
            print("Player Training Started!")
            _, _, self._ideal_max_number = self.player.train()
            self._freeze = False
            print(f"Player Training Ended. Stats: Random Difficulty: {self.difficulty}"
                  f"Lives Lost {self._ideal_lives_lost}, "
                  f"Max To Appear {self._ideal_max_number}.")
            self.current_iteration = 0

        self._seed = random.randint(0, 2 ** 32 - 1)
        self._level = gen_level(self.size, self._seed)
        self._changes_made = 0
        self._steps_taken = 0
        self.current_prime_count = 0
        self._change_map = np.zeros((self.size, self.size))

        observation = self.get_observation()
        return observation, {}

    def step(self, action):
        rewards = 0
        lvl_board = None
        eaten_numbers = None
        if self._changes_made <= self._max_changes:
            change, new_value = self.update(action, self._changes_made)
            if change:
                self._level[self._changes_made // self.size][self._changes_made % self.size] = new_value
                self.current_number = new_value

                min_number = self._ideal_max_number - (self._accepted_max_variation * 2)
                max_number = self._ideal_max_number + self._accepted_max_variation
                if sympy.isprime(self.current_number) and (max_number >= self.current_number >= min_number):
                    self.current_prime_count += 1

                self._changes_made += 1
                rewards += \
                    calculate_rewards_with_in_range_method(self.current_number, self._ideal_max_number,
                                                           self._accepted_max_variation, 0.1) + \
                    calculate_rewards_with_in_range_method(self.current_prime_count, self._ideal_prime_count,
                                                           self._accepted_prime_count_variations, 0.2)

        if self.is_playable() and self._changes_made == self._max_changes and self.is_training:
            print(f"Current Prime Count: {self.current_prime_count}")
            rewards += 10
            preset = self.get_preset_level()
            self.current_lost_lives, self.remaining_primes, lvl_board, eaten_numbers = self.player.play(preset,
                                                                                                        default=not
                                                                                                        self.is_training
                                                                                                        )

            rewards += \
                calculate_rewards_with_in_range_method(self.current_lost_lives, self._ideal_lives_lost,
                                                       self._accepted_lives_variation, 0.5)

        elif not self.is_playable() and self._changes_made == self._max_changes:
            rewards -= 20

        self._steps_taken += 1
        self.current_iteration += 5

        observation = self.get_observation()
        done = self._changes_made == self._max_changes or self._steps_taken >= self._max_steps

        if done and self.is_training:
            self.render(lvl_board, eaten_numbers)

        return observation, rewards, done, False, {}

    def update(self, action, cords):
        new_value = action + 1
        change = True if self._level[cords // self.size][cords % self.size] != action else False
        return change, new_value

    def get_preset_level(self):
        min_num = self._ideal_max_number - self._accepted_max_variation
        max_num = self._ideal_max_number + self._accepted_max_variation
        primes, prime_count = check_primes(self._level, min_num, max_num)
        return self._level.copy(), primes, prime_count

    def is_playable(self):
        if 2 <= self.current_prime_count <= 9:
            self._playable = 1
            return True
        else:
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

        print(f'Variations: {self._get_variations()}')

    def _get_variations(self):
        return {'Lives': [self.current_lost_lives, self._ideal_lives_lost],
                'Primes': [self.current_prime_count - self.remaining_primes, self.current_prime_count],
                'Max': [max(self._level.flatten()), self._ideal_max_number]}

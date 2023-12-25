import random
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from Environments.Generator.helper import gen_level, calculate_rewards_PCGRL, calculate_rewards_with_in_range_method, \
    check_primes_and_max


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

        self._stats = None
        self._ideal_lives_lost = 0
        self._ideal_time_spent = 0
        self._ideal_max_number = 0
        self._accepted_lives_variation = 1
        self._accepted_time_variation = 0.5  # In Min not Seconds
        self._accepted_max_variation = 5
        self._prob = {"prime": 0.3, "number": 0.7, "enemies": 0.0}  # Enemies are represented via 0

        self._changes_made = 0
        self._steps_taken = 0
        self._max_changes = int((self.size * self.size) * 0.5)
        self._max_steps = self._max_changes * self.size * self.size
        self._change_map = np.zeros((self.size, self.size))

        self._freeze = True

        # Agent related details
        self.current_number = 0
        self.current_lives = 0
        self.current_time = 0
        self.old_number = 0
        self.old_lives = 0
        self.old_time = 0

        self.observation_space = spaces.Dict({
            "map": spaces.Box(low=-1, high=self.max_num, shape=(self.size, self.size), dtype=int),
            "change_map": spaces.Box(low=0, high=self._max_changes, dtype=int, shape=(self.size, self.size)),
            "ideal_max": spaces.Box(low=0, high=self.max_num, dtype=int, shape=(1,)),
            "ideal_time": spaces.Box(low=0, high=30, dtype=float, shape=(1,)),
            "ideal_lives": spaces.Box(low=0, high=10, dtype=float, shape=(1,)),
            "current_max": spaces.Box(low=0, high=self.max_num, dtype=int, shape=(1,)),
            "current_lives": spaces.Box(low=0, high=10, dtype=float, shape=(1,)),
            "current_time": spaces.Box(low=0, high=30, dtype=float, shape=(1,)),
            "old_max": spaces.Box(low=0, high=self.max_num, dtype=int, shape=(1,)),
            "old_lives": spaces.Box(low=0, high=10, dtype=float, shape=(1,)),
            "old_time": spaces.Box(low=0, high=30, dtype=float, shape=(1,))
        })

        # Using Wide Representation
        self.action_space = spaces.MultiDiscrete([self.size, self.size, self.max_num + 1])

    def get_observation(self):
        return {
            "map": self._level.copy(),
            "change_map": self._change_map.copy(),
            "ideal_max": self._ideal_max_number,
            "ideal_time": self._ideal_time_spent,
            "ideal_lives": self._ideal_lives_lost,
            "current_max": self.current_number,
            "current_lives": self.current_lives,
            "current_time": self.current_time,
            "old_max": self.old_number,
            "old_lives": self.old_lives,
            "old_time": self.old_time,
        }

    def reset(self, seed=None, options=None):
        if self.current_iteration >= self.max_iterations:
            self._freeze = True

        if self._freeze:
            print("Player Training Started!")
            average_lives_lost, average_time_taken, highest_number = self.player.train()
            self._ideal_lives_lost = average_lives_lost
            self._ideal_time_spent = average_time_taken
            self._ideal_max_number = highest_number
            self._freeze = False
            self._seed = random.randint(0, 2 ** 32 - 1)
            self._past_seeds.append(self._seed)
            print(f"Player Training Ended. Stats: Lives Lost {self._ideal_lives_lost}, "
                  f"Time Cost {self._ideal_time_spent}, "
                  f"Max To Appear {self._ideal_max_number}.")
            self.current_iteration = 0
            # self.count = 0

        # self.count += 1
        # if self._level is not None and self.count == 100:
        #     self.render()
        #     self.count = 0

        if self._level is None:
            self._level, _, _ = gen_level(self.size, self._prob, self._seed)

        self._changes_made = 0
        self._steps_taken = 0
        self._change_map = np.zeros((self.size, self.size))
        self.current_iteration += 1

        observation = self.get_observation()
        return observation, {}

    def step(self, action):
        self._steps_taken += 1
        print(f'Step: {self._steps_taken}')
        self.old_lives, self.old_time, self.old_number = self.current_lives, self.current_time, self.current_number
        old_level = self._level.copy()
        change, y, x = self.update(action)
        if change > 0:
            self._changes_made += 1
            self._change_map[y][x] += 1
            self.current_number = action[2]
            primes, max_number = check_primes_and_max(self._level)
            self.current_lives, self.current_time, _ = self.player.play(self._level.copy(), primes, max_number)

        observation = self.get_observation()
        self.render(old_level)

        if self.old_lives != 0:
            rewards = calculate_rewards_PCGRL(self.current_lives, self.old_lives, self._ideal_lives_lost,
                                              self._accepted_lives_variation) + \
                      calculate_rewards_PCGRL(self.current_time, self.old_time, self._ideal_time_spent,
                                              self._accepted_time_variation) + \
                      calculate_rewards_PCGRL(self.current_number, self.old_number, self._ideal_max_number,
                                              self._accepted_max_variation)
        else:
            rewards = 0

        # rewards = calculate_rewards_with_in_range_method(self._stats["lives"],self._ideal_lives_lost,
        #                                                  self._accepted_lives_variation, 3)\
        #           + calculate_rewards_with_in_range_method(self._stats["time"],self._ideal_time_spent,
        #                                                    self._accepted_time_variation, 1) \
        #           + calculate_rewards_with_in_range_method(self._stats["max_num"],self._ideal_max_number,
        #                                                    self._accepted_max_variation, 5)

        done = self._changes_made >= self._max_changes or self._steps_taken >= self._max_steps
        # info = {
        #     ["Iteration"]: self._steps_taken,
        #     ["changes"]: self._changes_made,
        #     ["Lives & Time Lost"]: [self._stats["lives"], self._stats["time"]],
        #     ["Ideal Values"]: [self._ideal_lives_lost, self._ideal_time_spent],
        #     ["Difference"]: [self._stats["lives"] - self._ideal_lives_lost,
        #                      self._stats['time'] - self._ideal_time_spent]
        # }
        # self.render()
        return observation, rewards, done, False, {}

    def update(self, action):
        change = [0, 1][self._level[action[1]][action[0]] != action[2]]  # Some nice and fancy shortened code
        self._level[action[1]][action[0]] = action[2]
        return change, action[0], action[1]

    def render(self, old):
        print(f'Current Iteration: {self.current_iteration}')
        print(f'Base Level: ')
        for row in old:
            print(row)

        print(f'Final Level: ')
        for row in self._level:
            print(row)

        print(f'Variations: (Lives, Time, Max)')
        for i in self._get_variations():
            print(i)

    def _get_variations(self):
        return [[self.current_lives, self._ideal_lives_lost], [self.current_time, self._ideal_time_spent],
                [max(self._level.flatten()), self._ideal_max_number]]



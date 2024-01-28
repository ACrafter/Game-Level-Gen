import time
from threading import Thread

import gymnasium as gym
import pygame
from gymnasium import spaces
import numpy as np
import sympy

from Environments.Player.helper import gen_board


class PlayerV3(gym.Env):
    def __init__(self, size=5, max_number=100, render_mode=None, training=False, preset_level=None):
        self.preset_level = preset_level
        self.render_mode = render_mode
        self.max_value = max_number
        self.training = training
        self.size = size

        self.current_max = 5
        self.max_lives = 10
        self.max_time = 30
        self.prime_percent = 0.3
        self.max_prime_count = int(self.prime_percent * self.size * self.size)
        self.num_actions = 5

        self.board, self._current_primes = gen_board(self.size, self.current_max, self.max_prime_count)
        self.lives = self.max_lives
        self.remaining_time = self.max_time
        self.remaining_primes = self.max_prime_count
        self.player_pos = np.random.randint(0, self.size), np.random.randint(0, self.size)
        self.current_number = self.board[self.player_pos[0]][self.player_pos[1]]
        self.on_a_prime = False

        # Initialize other attributes
        self.start_time = time.time()  # Record the starting time
        self.steps = 0

        # Set up a thread for rendering
        if render_mode == 'Human':
            # Initialize Pygame
            pygame.init()
            self.screen_size = (800, 800)
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption('Number Muncher')
            # self.render_thread = Thread(target=self.render_thread_function)
            # self.render_thread.daemon = True  # Terminate thread when the main program exits
        else:
            self.agent_moves = []
            self.agent_actions = []
            self.average_time = []
            self.average_lives = []
            self.rewards = {"Move": [], "Eat": [], "Progress": []}
            self.numbers_eaten = []
            self.max_numbers = []

        self.observation_space = spaces.Dict({
            'grid_state': spaces.Box(low=0, high=self.max_value, shape=(25,), dtype=int),
            'player_position': spaces.Box(low=0, high=self.size - 1, shape=(2,), dtype=int),
            'nearest_prime_distance': spaces.Box(low=0, high=np.sqrt(2) * self.size, shape=(1,),
                                                 dtype=np.float32),
            'current_number_is_prime': spaces.Box(shape=(1,), dtype=bool, low=0, high=1),
            'nearest_prime_cords': spaces.Box(low=0, high=4, shape=(2,), dtype=int),
            'prime_density': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'direction_info': spaces.Box(low=0, high=1, shape=(2,), dtype=int),
            'immediate_reward': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'intermediate_reward': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'prime_eating_reward': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),

            'time_remaining': spaces.Box(low=0, high=self.max_time, shape=(1,), dtype=int),
            'lives_remaining': spaces.Box(low=0, high=self.max_lives, shape=(1,), dtype=int),
            'prime_numbers_remaining': spaces.Box(low=0, high=self.max_prime_count, shape=(1,), dtype=int),
            'current_number': spaces.Box(low=0, high=self.max_value, shape=(1,), dtype=int),
        })

        self.observation = None
        self.action_space = spaces.Discrete(5)

        # Gen info
        self.total_levels_played = -1
        self.total_lives_lost = 0
        self.total_time_spend = np.inf
        self.max_num_appeared = 0

    def _get_eval_observations(self):
        return {
            'grid_state': np.ravel(self.board).reshape(25, ),
            'player_position': np.array(self.player_pos),
            'lives_remaining': np.array([self.lives]),
            'prime_numbers_remaining': np.array([self.remaining_primes]),
            'current_number': np.array([self.current_number]),
            'current_number_is_prime': np.array([self.on_a_prime]),
            'time_remaining': np.array([self.remaining_time]),
            'nearest_prime_distance': np.array([0]),
            'nearest_prime_cords': np.array([0, 0]),
            'prime_density': np.array([0]),
            'direction_info': np.zeros(2),
            'immediate_reward': np.array([0]),
            'intermediate_reward': np.array([0]),
            'prime_eating_reward': np.array([0]),
        }

    def reset(self, seed=None, options=None):
        # Logging Stuff for the Generator
        self.total_levels_played += 1
        self.total_time_spend = abs(min(self.remaining_time, self.total_time_spend))
        self.start_time = time.time()
        if self.preset_level is None:
            self.board, self._current_primes = gen_board(self.size, self.current_max, self.max_prime_count)
            self.remaining_primes = self.max_prime_count
        else:
            self.board, self._current_primes, self.remaining_primes = self.preset_level
        self.lives = self.max_lives
        self.remaining_time = self.max_time
        self.player_pos = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
        self.current_number = self.board[self.player_pos[0]][self.player_pos[1]]

        if sympy.isprime(self.current_number):
            self.on_a_prime = True
        else:
            self.on_a_prime = False

        self.observation = self._get_eval_observations()

        return self.observation, {}

    def step(self, action):
        self.steps += 1
        old_prime_distance, _ = self.calculate_nearest_prime_distance()
        move_direction = [0, 0]
        move_reward = 0
        eat_reward = 0
        progression_reward = 0

        if action < 4:
            new_position, moved, move_direction = self.calculate_direction_info(action)
            new_prime_distance, _ = self.calculate_nearest_prime_distance()
            move_reward += self.calculate_movement_reward(moved, old_prime_distance, new_prime_distance)
        else:
            eat_reward += self.calculate_eat_reward()
            progression_reward += self.calculate_progression_reward()

        new_prime_distance, new_prime_cords = self.calculate_nearest_prime_distance()
        prime_density = self.calculate_prime_density()
        self.remaining_time -= int((time.time() - self.start_time))

        new_observation = self._get_eval_observations()
        new_observation['nearest_prime_distance'] = np.array([new_prime_distance])
        new_observation['nearest_prime_cords'] = np.array(new_prime_cords)
        new_observation['prime_density'] = np.array([prime_density])
        new_observation['direction_info'] = np.array(move_direction)
        new_observation['immediate_reward'] = np.array([move_reward])
        new_observation['intermediate_reward'] = np.array([progression_reward])
        new_observation['prime_eating_reward'] = np.array([eat_reward])

        total_reward = move_reward + progression_reward + eat_reward
        done = True if self.lives == 0 or self.remaining_primes == 0 or self.remaining_time == 0 else False

        if self.lives == 0:
            self.current_max = 5

        if self.remaining_primes == 0 and self.current_max < self.max_value:
            self.current_max += 5

        self.max_num_appeared = max(self.max_num_appeared, max(self.board.flatten()))

        # if self.preset_level:
        #     self.max_num_appeared = max(self.board.flatten())

        info = {
            'previous_action': action,
            'nearest_prime_distance': new_prime_distance,
            'nearest_prime_cords': new_prime_cords,
            'prime_density': prime_density,
            'direction_info': move_direction,
            'immediate_reward': move_reward,
            'intermediate_reward': progression_reward,
            'prime_eating_reward': eat_reward
        }

        if self.render_mode != 'Human' and 49_999 <= self.steps:
            self.agent_actions.append(action)
            self.agent_moves.append(move_direction)
            self.average_lives.append(self.lives)
            self.max_numbers.append(self.current_max)

            if self.steps >= 50_000:
                self.render()
                self.steps = 0
                self.agent_moves = []
                self.agent_actions = []
                self.average_lives = []
                self.max_numbers = []
                self.numbers_eaten = []

        if self.render_mode == 'Human':
            clock = pygame.time.Clock()
            self.render()
            pygame.display.flip()
            clock.tick(30)

        return new_observation, total_reward, done, False, info

    def calculate_nearest_prime_distance(self):
        player_position = self.observation['player_position']
        primes_positions = np.argwhere(np.isin(self.board, self._current_primes))
        if len(primes_positions) == 0:
            return np.sqrt(2) * self.size, [-1, -1]  # Return max distance if no primes on the board

        distances = np.linalg.norm(primes_positions - player_position, axis=1)
        cords = abs(primes_positions - player_position)

        if cords.size == 1:
            cords = cords[0]
        else:
            cords = min(cords, key=lambda arr: np.sum(arr))

        if distances.size == 1:
            distances = distances[0]
        else:
            distances = np.min(distances)

        return distances, cords

    def calculate_prime_density(self):
        # Calculate the density of prime numbers in the vicinity of the player
        player_position = self.observation['player_position']
        prime_density = 0.0
        radius = 2  # Define the radius of the vicinity (adjust as needed)
        for i in range(max(0, player_position[0] - radius), min(self.size, player_position[0] + radius + 1)):
            for j in range(max(0, player_position[1] - radius),
                           min(self.size, player_position[1] + radius + 1)):
                if self.board[i, j] in self._current_primes:
                    prime_density += 1

        return prime_density / ((2 * radius + 1) ** 2)

    def calculate_direction_info(self, action):
        directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        new_position = self.player_pos + directions[action]
        if 0 <= new_position[0] <= 4 and 0 <= new_position[1] <= 4:
            self.player_pos += directions[action]
            self.current_number = self.board[self.player_pos[0]][self.player_pos[1]]

            if sympy.isprime(self.current_number):
                self.on_a_prime = True

            return new_position, True, directions[action]
        else:
            return new_position, False, directions[action]

    def calculate_movement_reward(self, moved, old_distance, new_distance):
        # Moving towards a prime or if the current number is prime
        if not moved:
            return -0.5

        if new_distance < old_distance:
            return 0.5

        if old_distance < new_distance:
            return -2

        return 0

    def calculate_eat_reward(self):
        if self.render_mode != 'Human':
            self.numbers_eaten.append(self.current_number)

        if sympy.isprime(self.current_number):
            self.on_a_prime = False
            self.remaining_primes -= 1
            self.current_number = self.board[self.player_pos[0]][self.player_pos[1]] = 0
            return + 5
        else:
            self.lives -= 1
            self.total_lives_lost += 1
            self.current_number = self.board[self.player_pos[0]][self.player_pos[1]] = 0
            return - 2

    def calculate_progression_reward(self):
        if self.max_prime_count - self.remaining_primes != 0:
            return (self.max_prime_count - self.remaining_primes) * 0.1
        else:
            return - 1

    # def render_thread_function(self):
    #     clock = pygame.time.Clock()
    #     running = True
    #     while running:
    #         for event in pygame.event.get():
    #             if event.type == pygame.QUIT:
    #                 running = False
    #
    #         self.render()
    #         time.sleep(1)
    #         pygame.display.flip()
    #         clock.tick(60)  # Adjust the frame rate as needed
    #
    # def start_render_thread(self):
    #     # Start the rendering thread
    #     self.render_thread.start()

    def render(self):
        if self.render_mode == 'Human':
            # Render the current state of the game
            self.screen.fill((255, 255, 255))  # White background

            # Draw the game elements using Pygame drawing functions
            self.draw_board()
            self.draw_agent()
            self.draw_info()
        else:
            print(f"Agent Actions: {self.agent_actions} \n"
                  f"Agent Steps: {self.agent_moves} \n "
                  f"Agent Lives: {self.average_lives} \n "
                  f"Numbers Eaten: {self.numbers_eaten} \n"
                  f"Maxes: {self.max_numbers}")

    def draw_board(self):
        # Draw the game board using rectangles
        cell_size = self.screen_size[0] // len(self.board)

        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                number = self.board[i, j]
                rect_color = (0, 0, 0) if number == 0 else (0, 255, 0) if number in self._current_primes else (
                    139, 0, 0)
                pygame.draw.rect(self.screen, rect_color, (j * cell_size, i * cell_size, cell_size, cell_size), 0)

                # Display the number on the rectangle
                font = pygame.font.Font(None, 24)
                number_text = font.render(str(number), True, (255, 255, 255))
                text_rect = number_text.get_rect(
                    center=(j * cell_size + cell_size // 2, i * cell_size + cell_size // 2))
                self.screen.blit(number_text, text_rect)

    def draw_agent(self):
        # Draw the agent as a circle or other shape
        agent_size = self.screen_size[0] // len(self.board)
        agent_position = (
            self.player_pos[1] * agent_size + agent_size // 3,
            self.player_pos[0] * agent_size + agent_size // 2
        )
        agent_radius = agent_size // 4  # Adjust the factor for the agent's size

        pygame.draw.circle(self.screen, (0, 0, 255), agent_position, agent_radius)

    def draw_info(self):
        # Draw remaining lives, remaining primes, and current max number
        font = pygame.font.Font(None, 36)

        # Update elapsed time
        # elapsed_time = (pygame.time.get_ticks() - self.start_time) // 1000  # in seconds
        # self.remaining_time -= elapsed_time
        time_text = font.render(f'Remaining Time: {self.remaining_time}s', True, (0, 0, 0))
        self.screen.blit(time_text, (10, self.screen_size[1] - 90))

        lives_text = font.render(f'Lives: {self.lives}', True, (0, 0, 0))
        self.screen.blit(lives_text, (10, self.screen_size[1] - 60))

        primes_text = font.render(f'Remaining Primes: {self.remaining_primes}', True, (0, 0, 0))
        self.screen.blit(primes_text, (10, self.screen_size[1] - 30))

    def close(self):
        # Close the Pygame window when done
        pygame.quit()

    def get_average_lives_lost(self):
        if self.total_levels_played > 0:
            return self.total_lives_lost // self.total_levels_played
        else:
            return 0

    def get_average_time_lost(self):
        if self.total_levels_played > 0:
            return self.total_time_spend
        else:
            return 0

    def get_max_number_appeared(self):
        return self.max_num_appeared

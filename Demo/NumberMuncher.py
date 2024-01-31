import time

import pygame
import numpy as np
import sympy
from stable_baselines3 import PPO

from Environments.Generator.GenV3 import Generator

''' what did I change?
1. the right and left events were switched.
2. the board was rendered transvered, I fixed it in
lines 103:106 I switched the x and y coordinate.
3. the missing column, the width was used to be divided by 5 cells
so there were no space to print the fifth column becasue we were printing from y+1
anyway I adjusted the cell size to be width / 6 to give us space.
4. adjusting the coordinates of timer, lives and level texts
somehow the blit function is switched, the columns and rows both are switched.
5. the right and down actions were switched. I switched them back!
'''

HEIGHT = 750
WIDTH = 650
MENU_WIDTH = 800
MENU_HEIGHT = 600
DIFFICULTY_LEVELS = ['Easy', 'Normal', 'Hard']
TIMER_DURATION = [300, 240, 180]


class Menu:
    def __init__(self, app):
        self.screen = pygame.display.set_mode((MENU_WIDTH, MENU_HEIGHT))
        pygame.display.set_caption("Main Menu")
        self.background = pygame.image.load(
            r"./assets/Munchers_Menu.png").convert_alpha()
        self.font = pygame.font.Font(
            r"./assets/Comfortaa-VariableFont_wght.ttf", 32)
        self.difficulty = 0
        self.on = True

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if 300 <= x <= 500 and 200 <= y <= 250:
                    # Play button clicked
                    self.on = False
                elif 300 <= x <= 500 and 300 <= y <= 350:
                    # Set Difficulty button clicked
                    self.difficulty = (self.difficulty + 1) % len(DIFFICULTY_LEVELS)
                elif 300 <= x <= 500 and 400 <= y <= 450:
                    # Exit button clicked
                    pygame.quit()
                    exit()

    def render(self):
        self.screen.blit(self.background, (0, 0))

        # Draw buttons
        button_width, button_height = 504, 64
        button_color = (255, 255, 255)

        play_button = self.font.render("Play", True, (0, 0, 0))
        pygame.draw.rect(self.screen, (0, 0, 0), (198, 210, button_width + 4, button_height + 4), border_radius=13)
        pygame.draw.rect(self.screen, button_color, (200, 212, button_width, button_height), border_radius=13)
        self.screen.blit(play_button, (450 - play_button.get_width() // 2, 240 - play_button.get_height() // 2))

        difficulty_button = self.font.render(f"Set Difficulty: {DIFFICULTY_LEVELS[self.difficulty]}", True,
                                             (0, 0, 0))

        pygame.draw.rect(self.screen, (0, 0, 0), (198, 310, button_width + 4, button_height + 4), border_radius=13)
        pygame.draw.rect(self.screen, button_color, (200, 312, button_width, button_height), border_radius=13)
        self.screen.blit(difficulty_button,
                         (450 - difficulty_button.get_width() // 2, 340 - difficulty_button.get_height() // 2))

        exit_button = self.font.render("Exit", True, (0, 0, 0))
        pygame.draw.rect(self.screen, (0, 0, 0), (198, 418, button_width + 4, button_height + 4), border_radius=13)
        pygame.draw.rect(self.screen, button_color, (200, 420, button_width, button_height), border_radius=13)
        self.screen.blit(exit_button, (450 - exit_button.get_width() // 2, 450 - exit_button.get_height() // 2))

        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()

        while self.on:
            self.render()
            self.handle_events()
            clock.tick(15)


class NumberMuncher:
    def __init__(self, gen_path=None, env=None):
        pygame.init()
        self.gen_path = gen_path
        self.env = env

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Number Muncher')
        self.font = pygame.font.Font(
            r"assets\Comfortaa-VariableFont_wght.ttf", 18)
        self.muncher = pygame.image.load('./assets/muncher_char.jpg')

        self._player_lives = 10
        self._player_position = [0, 0]
        self.cell_size = 115
        self.level = 1
        self.score = 0
        self.muncher = pygame.transform.scale(self.muncher, (50, 50))

        self.start_time = None
        self.max_time = None
        self._difficulty = None
        self.screen = None
        self.game_over = False

        self.board = []
        self.remaining_primes = 10
        self.main_menu()

    def main_menu(self):
        m = Menu(self)
        m.run()
        self._difficulty = m.difficulty
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.setup()

    def setup(self):
        """
        Sets up the generator environment with the proper difficulty setting and activates the generator
        :return: None
        """
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Number Muncher")
        if self.gen_path and self.env:
            self.env.difficulty = self._difficulty
            Gen_model = PPO.load(self.gen_path, self.env)

            self.env.reset()
            self.env.current_lost_lives = 10 - self._player_lives
            done = False
            playable = False
            while not done:
                action, _ = Gen_model.predict(self.env.get_observation())
                state, reward, done, _, _ = self.env.step(action)
                playable = self.env.is_playable()

            print(f"Playable: {self.env.is_playable()}")
            self.board, _, self.remaining_primes = self.env.get_preset_level()
        else:
            self.board = np.random.randint(0, 100, size=(5, 5))
            for i in self.board.flatten():
                if sympy.isprime(i):
                    self.remaining_primes += 1

        self.start_time = pygame.time.get_ticks() // 1000
        self.max_time = TIMER_DURATION[1]

        print(self.remaining_primes)
        self.run()

    def handle_events(self):

        if self.remaining_primes == 0:
            self.setup()

        if self._player_lives == 0:
            self.game_over = True
            print('Game Over')

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.KEYDOWN:
                keys = pygame.key.get_pressed()

                # Check if the up arrow is pressed
                if keys[pygame.K_UP]:
                    pygame.draw.rect(self.screen, (0, 0, 0), (self.cell_size * self._player_position[1] + 50,
                                                              self.cell_size * self._player_position[0] + 50,
                                                              self.cell_size,
                                                              self.cell_size))
                    self._player_position[0] -= 1 if self._player_position[0] > 0 else self._player_position[0]

                if keys[pygame.K_DOWN]:
                    pygame.draw.rect(self.screen, (0, 0, 0), (self.cell_size * self._player_position[1] + 50,
                                                              self.cell_size * self._player_position[0] + 50,
                                                              self.cell_size,
                                                              self.cell_size))
                    self._player_position[0] += 1 if self._player_position[0] < 4 else self._player_position[0]

                # Check if the left arrow is pressed
                if keys[pygame.K_RIGHT]:
                    pygame.draw.rect(self.screen, (0, 0, 0), (self.cell_size * self._player_position[1] + 50,
                                                              self.cell_size * self._player_position[0] + 50,
                                                              self.cell_size,
                                                              self.cell_size))
                    self._player_position[1] += 1 if self._player_position[1] < 4 else self._player_position[1]

                # Check if the right arrow is pressed
                if keys[pygame.K_LEFT]:
                    pygame.draw.rect(self.screen, (0, 0, 0), (self.cell_size * self._player_position[1] + 50,
                                                              self.cell_size * self._player_position[0] + 50,
                                                              self.cell_size,
                                                              self.cell_size))
                    self._player_position[1] -= 1 if self._player_position[1] > 0 else self._player_position[1]

                if keys[pygame.K_SPACE]:
                    current_number = self.board[self._player_position[0]][self._player_position[1]]
                    if sympy.isprime(current_number):
                        self.remaining_primes -= 1
                        self.score += 1
                    else:
                        self._player_lives -= 1

                    print(f'Score: {self.score}, Primes to eat: {self.remaining_primes}')
                    self.board[self._player_position[1]][self._player_position[0]] = 0

    def render(self):
        # Draw background
        background = pygame.image.load('./assets/Game_Background.png')
        self.screen.blit(background, (0, 0))

        # Draw the level
        level_text = self.font.render(f'Level: {self.level}', True, (56, 12, 89))
        self.screen.blit(level_text, (450 - level_text.get_width() // 2, 240 - level_text.get_height() // 2))

        # Draw the board
        x, y = 0, 0
        for row in self.board:
            x, y = x + 1, 0
            for e in row:
                y += 1
                # pygame.draw.rect(self.screen, (0, 0, 0), (self.cell_size * y, self.cell_size * x, self.cell_size,
                #                                           self.cell_size))
                num = self.font.render(f'{e}', True, (255, 255, 255))
                current_dims = (self.cell_size * x, self.cell_size * y + 50)
                self.screen.blit(num, current_dims)

        # Draw the player
        self.screen.blit(self.muncher, (self.cell_size * self._player_position[1] + 50,
                                        self.cell_size * self._player_position[0] + 150))

        # Draw the timer
        current_time = pygame.time.get_ticks() // 1000  # Current time in seconds
        elapsed_time = current_time - self.start_time
        remaining_time = max(self.max_time - elapsed_time, 0)

        timer_text = self.font.render(f"Time: {remaining_time // 60}:{remaining_time % 60}", True,
                                      (255, 255, 255), (255, 0, 0))
        self.screen.blit(timer_text, (500, 650))

        # Draw the level
        level_text = self.font.render(f"Level: {self.level}", True,
                                      (255, 255, 255), (255, 0, 0))
        self.screen.blit(level_text, (500, 50))

        # Draw the lives
        lives_text = self.font.render(f"Lives: {self._player_lives}", True,
                                      (255, 255, 255), (255, 0, 0))
        self.screen.blit(lives_text, (50, 50))

        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        while not self.game_over:
            self.handle_events()
            self.render()
            clock.tick(30)

        if self.game_over:
            self.setup()
            self.game_over = False


N = NumberMuncher('../Final_Generator_models/V3/GenV3 -- Iteration#4.zip', Generator(False, None, 100))

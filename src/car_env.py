import os
import random
import sys

import numpy as np
import pygame
from gymnasium import Env
from gymnasium.spaces import Discrete, MultiDiscrete

from .sprites import Car, Obstacle, Target


class CarEnv(Env):
    def __init__(
        self,
        seed=None,
        training=True,
        obstacles_count=5,
        grid=True,
    ):
        screen_resolution = (800, 800)
        self.block_size = 100

        pygame.init()
        self.screen = pygame.display.set_mode(screen_resolution)
        self.clock = pygame.time.Clock()

        pygame.display.set_caption("Car Reinforcement Learning")

        font_path = os.path.join("assets", "fonts", "Mario-Kart-DS.ttf")
        self.font = pygame.font.Font(font_path, 50)

        self.progress = 0  # 0%
        self.truncation_limit = 30
        self.truncation = self.truncation_limit

        # 0 - Left, 1 - Right, 2 - Top, 3 - Bottom
        self.action_space = Discrete(4)
        self.observation_space = MultiDiscrete([3] * 24 + [30 + 1])

        self.done = False

        # stores all locations already visited, used to promote exploring
        self.memory = set()

        self.level = random.randint(0, 999)

        if seed:
            random.seed(None)
            self.level = seed

        self.car = Car(env=self, block_size=100)
        self.target = Target(env=self)
        self.obstacle = Obstacle(env=self, spawns=obstacles_count, block_size=100)

        self.memory.add(self.car.rect.center)

        self.background = None
        if not training:
            self.background = self.draw_background()

        self.grid = grid

    def draw_background(self):
        background_image_path = os.path.join("assets", "images", "bg-tile.png")
        background_image = pygame.image.load(background_image_path)
        background_image = pygame.transform.scale(
            background_image, (self.block_size // 2, self.block_size // 2)
        )
        background = pygame.Surface((self.screen.get_width(), self.screen.get_height()))

        for x in range(0, self.screen.get_width(), self.block_size // 2):
            for y in range(0, self.screen.get_height(), self.block_size // 2):
                background.blit(background_image, (x, y))

        return background

    def draw_grid(self):
        white = (0, 0, 0)
        for x in range(0, self.screen.get_width(), self.block_size):
            for y in range(0, self.screen.get_height(), self.block_size):
                rect = pygame.Rect(x, y, self.block_size, self.block_size)
                pygame.draw.rect(self.screen, white, rect, 1)

    def step(self, action):
        self.truncation -= 1
        self.car.time_driving += 1
        reward = 0

        if action == 0:
            result = self.car.move_left()
        elif action == 1:
            result = self.car.move_right()
        elif action == 2:
            result = self.car.move_up()
        elif action == 3:
            result = self.car.move_down()
        if not result:
            reward = -3
            self.done = True

        # Promotes exploring
        if self.car.rect.center in self.memory:
            reward = -0.2

        self.memory.add(self.car.rect.center)

        # Check collision with obstacles using the sprite group
        if pygame.sprite.spritecollide(self.car, self.obstacle.obstacles, False):
            self.done = True
            reward = -3

        elif pygame.sprite.spritecollide(self.car, [self.target], False):
            reward = 5
            self.done = True

        else:
            if self.truncation < 0:
                reward = -3
                self.done = True
                self.truncation = 0

        observation = self._get_obs()

        return (
            observation,
            reward,
            self.done,
            self.truncation <= 0,
            {
                "position": self.car.rect.center,
                "score": reward,
                "done": self.done,
            },
        )

    def map_to_grid(self, position):
        x, y = position
        x = (x - 50) // 100
        y = (y - 50) // 100
        return x, y

    def _get_obs(self):
        car_x, car_y = self.car.rect.center
        obstacle_positions = [
            obstacle.rect.center for obstacle in self.obstacle.obstacles
        ]
        # fmt: off
        # Define the surrounding positions two moves ahead
        surrounding_positions = [
                                                    (+  0, -300),                                                  # First row
                                      (-100, -200), (+  0, -200), (+100, -200),                                    # Second row
                        (-200, -100), (-100, -100), (+  0, -100), (+100, -100), (+200, -100),                      # Third row
          (-300, +  0), (-200, +  0), (-100, +  0),               (+100, +  0), (+200, +  0), (+300, +  0),        # Middle row
                        (-200, +100), (-100, +100), (+  0, +100), (+100, +100), (+200, +100),                      # -Third row
                                      (-100, +200), (+  0, +200), (+100, +200),                                    # -Second row
                                                    (+  0, +300)                                                   # -First row
        ]
        # fmt: on

        # Initialize the observation with zeros
        observation = np.zeros(shape=(25,), dtype=np.int64)
        screen_width, screen_height = self.screen.get_width(), self.screen.get_height()

        # Map surroundings to the grid
        for i, (x, y) in enumerate(surrounding_positions):
            x, y = car_x + x, car_y + y
            if 0 <= x < screen_width and 0 <= y < screen_height:
                if (x, y) == self.target.rect.center:
                    observation[i] = 2  # Target
                elif (x, y) in obstacle_positions:
                    observation[i] = 1  # Obstacle
            else:
                observation[i] = 1  # Out of bounds ie: Edge, Obstacle

        observation[-1] = self.truncation
        return observation

    def render(self):
        if self.background:
            self.screen.blit(self.background, (0, 0))
        else:
            self.screen.fill((255, 255, 255))

        if self.grid:
            self.draw_grid()

        self.car.render()
        self.target.render()
        self.obstacle.render()

        text_surface = self.font.render(f"Seed  {self.level}", True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(600, 50))
        self.screen.blit(text_surface, text_rect)

        pygame.display.flip()
        self.clock.tick(60)

    def reset(self, seed=None):
        self.done = False
        self.truncation = self.truncation_limit
        self.memory = set()

        self.level = random.randint(0, 999)
        if seed:
            random.seed(seed)
            self.level = seed

        self.car.reset()
        self.memory.add(self.car.rect.center)

        self.target.reset()
        self.obstacle.reset()
        return (self._get_obs(), {})

    def close(self):
        pygame.quit()
        sys.exit()

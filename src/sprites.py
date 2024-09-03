import os
import random

import numpy as np
import pygame

from . import helpers


class Car(pygame.sprite.Sprite):
    def __init__(
        self,
        env,
        block_size=100,
    ):
        super().__init__()
        self.env = env
        self.screen = env.screen

        # Load and scale the car image
        car_image_path = os.path.join("assets", "images", "car.png")
        self.original_image = pygame.image.load(car_image_path).convert_alpha()
        self.original_image = helpers.aspect_scale_image(self.original_image, 75)
        self.image = self.original_image.copy()

        # Set initial position and movement parameters
        self.position = (50, 50)
        self.block_size = block_size

        self.time_driving = 0

        # Car's rect (sprite's rect)
        self.rect = self.image.get_rect(center=self.position)

    def reset(self, seed=None):
        self.time_driving = 0
        self.position = (50, 50)
        self.rect = self.image.get_rect(center=self.position)

    def move_right(self):
        self.rect.centerx += self.block_size
        self.image, self.rect = helpers.rotate_center(
            self.original_image, 0, self.rect.center
        )
        if self.rect.centerx < self.screen.get_width():
            return True
        return False

    def move_left(self):
        self.rect.centerx -= self.block_size
        self.image, self.rect = helpers.rotate_center(
            self.original_image, 180, self.rect.center
        )
        if self.rect.centerx >= self.block_size:
            return True
        return False

    def move_up(self):
        self.rect.centery -= self.block_size
        self.image, self.rect = helpers.rotate_center(
            self.original_image, 90, self.rect.center
        )

        if self.rect.centery >= self.block_size:
            return True
        return False

    def move_down(self):
        self.rect.centery += self.block_size
        self.image, self.rect = helpers.rotate_center(
            self.original_image, -90, self.rect.center
        )

        if self.rect.centery < self.screen.get_height():
            return True
        return False

    def render(self):
        # Render the car on the screen
        self.screen.blit(self.image, self.rect.topleft)


class Target(pygame.sprite.Sprite):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.screen = env.screen

        # Load and scale the target image
        target_image_path = os.path.join("assets", "images", "flag.png")
        self.image = pygame.image.load(target_image_path).convert_alpha()
        self.image = helpers.aspect_scale_image(self.image, 70)

    def reset(self, seed=None):
        if seed:
            random.seed(seed)

        self.position = (self.screen.get_width() - 50, self.screen.get_height() - 50)
        self.rect = self.image.get_rect(center=self.position)

    def render(self):
        # Render the target on the screen
        self.screen.blit(self.image, self.rect.topleft)


class Obstacle(pygame.sprite.Sprite):
    def __init__(
        self,
        env,
        spawns=5,
        block_size=100,
    ):
        super().__init__()  # Initialize the pygame.sprite.Sprite
        self.env = env
        self.screen = env.screen
        self.spawns = spawns
        self.block_size = block_size

        self.images = list(self.load_images())

        self.grid = np.zeros(
            shape=(
                self.screen.get_width() // self.block_size,
                self.screen.get_height() // self.block_size,
            ),
            dtype=np.int16,
        )

    def reset(self, seed=None):
        if seed:
            random.seed(seed)

        self.obstacles = pygame.sprite.Group()
        self.grid = np.zeros_like(self.grid)
        for _ in range(self.spawns):
            self.obstacles.add(self.create_obstacle())

    def is_path_exists(self, matrix):
        rows, cols = matrix.shape
        visited = np.zeros_like(matrix, dtype=bool)

        def dfs(r, c):
            if r < 0 or c < 0 or r >= rows or c >= cols:  # Out of bounds
                return False
            if visited[r, c] or matrix[r, c] != 0:  # Already visited or not a 0
                return False
            if r == rows - 1 and c == cols - 1:  # Reached bottom-right corner
                return True

            visited[r, c] = True

            # Explore neighbors (up, down, left, right)
            return dfs(r + 1, c) or dfs(r - 1, c) or dfs(r, c + 1) or dfs(r, c - 1)

        return dfs(0, 0)

    def _set_obstacle_position(self, obstacle_sprite):
        # Randomly position the obstacle, ensuring no overlap.

        # Can be replaced with while True, this is used to prevent
        # freezing when obstacle can't be placed anywhere
        for _ in range(3_00_000):
            x_position, y_position = helpers.generate_random_grid_position(
                self.screen, self.block_size
            )
            grid_x, grid_y = (
                x_position // self.block_size,
                y_position // self.block_size,
            )

            temp_grid = self.grid.copy()
            temp_grid[grid_x][grid_y] = 3

            if self.grid[grid_x][grid_y] == 3:
                continue
            elif grid_x == grid_y and grid_x in [0, len(self.grid) - 1]:
                continue
            elif not self.is_path_exists(matrix=temp_grid):
                continue
            else:
                obstacle_sprite.rect.center = (x_position, y_position)
                self.grid[grid_x][grid_y] = 3
                break

        return obstacle_sprite

    def render(self):
        # Draw all obstacles on the screen
        self.obstacles.draw(self.screen)

    def load_images(self):
        image_path = os.path.join("assets", "images", "obstacles")
        for _file in os.listdir(image_path):
            if _file.endswith(".png"):
                image = pygame.image.load(
                    os.path.join(image_path, _file)
                ).convert_alpha()
                yield pygame.transform.scale(
                    image, (self.block_size // 2, self.block_size // 2)
                )

    def create_obstacle(self):
        obstacle_sprite = pygame.sprite.Sprite()  # Create a new sprite
        obstacle_sprite.image = random.choice(self.images)  # Set the image
        obstacle_sprite.rect = obstacle_sprite.image.get_rect()
        obstacle_sprite.moving = False
        return self._set_obstacle_position(obstacle_sprite)

import contextlib
import numpy as np
import pygame
import os
import sys
import math
import helpers
import random
from gymnasium.spaces import Box, Discrete
from stable_baselines3.common.env_checker import check_env
from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import sys
import gymnasium as gym


class Car(pygame.sprite.Sprite):
    def __init__(
        self,
        env,
        car_path=None,
        block_size=100,
        seed=None,
    ):
        super().__init__()
        self.env = env
        self.screen = env.screen

        # Load and scale the car image
        self.original_image = pygame.image.load(car_path).convert_alpha()
        self.original_image = helpers.aspect_scale_image(self.original_image, 75)
        self.image = self.original_image.copy()

        # Set initial position and movement parameters
        self.position = (50, 50)
        self.block_size = block_size

        self.time_driving = 0

        # Car's rect (sprite's rect)
        self.rect = self.image.get_rect(center=self.position)

    def reset(self, seed=None):
        if seed:
            random.seed(seed)

        self.time_driving = 0
        self.position = (50, 50)
        self.rect = self.image.get_rect(center=self.position)

    def move_right(self):
        if self.rect.centerx < self.screen.get_width() - self.block_size:
            self.rect.centerx += self.block_size
            self.image, self.rect = helpers.rotate_center(
                self.original_image, 0, self.rect.center
            )
            return True
        return False

    def move_left(self):
        if self.rect.centerx > self.block_size:
            self.rect.centerx -= self.block_size
            self.image, self.rect = helpers.rotate_center(
                self.original_image, 180, self.rect.center
            )
            return True
        return False

    def move_up(self):
        if self.rect.centery > self.block_size:
            self.rect.centery -= self.block_size
            self.image, self.rect = helpers.rotate_center(
                self.original_image, 90, self.rect.center
            )
            return True
        return False

    def move_down(self):
        if self.rect.centery < self.screen.get_height() - self.block_size:
            self.rect.centery += self.block_size
            self.image, self.rect = helpers.rotate_center(
                self.original_image, -90, self.rect.center
            )
            return True
        return False

    def render(self):
        # Render the car on the screen
        self.screen.blit(self.image, self.rect.topleft)
        # pygame.draw.rect(self.screen, (0, 255, 0), self.rect, 2)


class Target(pygame.sprite.Sprite):
    def __init__(
        self,
        env,
        target_path=None,
    ):
        super().__init__()  # Initialize the pygame.sprite.Sprite
        self.env = env
        self.screen = env.screen

        # Load and scale the target image
        self.image = pygame.image.load(target_path).convert_alpha()
        self.image = helpers.aspect_scale_image(self.image, 70)

        # Set the initial position
        self.update()

    def reset(self, seed=None):
        if seed:
            random.seed(seed)

        self.update()

    def update(self):
        # Set target's position and update its rect
        self.position = (self.screen.get_width() - 50, self.screen.get_height() - 50)
        self.rect = self.image.get_rect(center=self.position)
        return self.position

    def render(self):
        # Render the target on the screen
        self.screen.blit(self.image, self.rect.topleft)


class Obstacle(pygame.sprite.Sprite):
    def __init__(
        self,
        env,
        image_path,
        spawns=5,
        block_size=100,
    ):
        super().__init__()  # Initialize the pygame.sprite.Sprite
        self.env = env
        self.screen = env.screen

        self.image = pygame.image.load(image_path).convert_alpha()
        self.image = helpers.aspect_scale_image(self.image, 45)

        self.spawns = spawns
        self.block_size = block_size
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
        # freezing when obstacle can be placed nowhere
        for _ in range(200):
            x_position, y_position = helpers.generate_random_grid_position(
                self.screen, self.block_size
            )
            grid_x, grid_y = (
                x_position // self.block_size,
                y_position // self.block_size,
            )

            temp_grid = self.grid.copy()
            temp_grid[grid_x][grid_y] = 3

            if any(
                obstacle_sprite.rect.colliderect(other_obstacle.rect)
                for other_obstacle in self.obstacles
                if other_obstacle != obstacle_sprite
            ):
                continue
            elif x_position == y_position and x_position in [750, 50]:
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

    def create_obstacle(self):
        obstacle_sprite = pygame.sprite.Sprite()  # Create a new sprite
        obstacle_sprite.image = self.image
        obstacle_sprite.rect = obstacle_sprite.image.get_rect()
        obstacle_sprite.moving = False
        return self._set_obstacle_position(obstacle_sprite)


class CarEnv(Env):
    def __init__(self, seed=None, render_mode="human", obstacles=5):
        screen_resolution = (800, 800)

        pygame.init()
        self.screen = pygame.display.set_mode(screen_resolution)
        self.clock = pygame.time.Clock()

        self.progress = 0  # 0%
        self.truncation_limit = 50
        self.truncation = self.truncation_limit

        # 0 - Left, 1 - Right, 2 - Top, 3 - Bottom
        self.action_space = Discrete(4)

        # 1 added to use CnnPolicy and simulate the channel dimension
        self.observation_space = Box(
            low=0, high=4, shape=(8, 8), dtype=np.int8
        )

        self.done = False

        if seed:
            random.seed(None)

        self.car = Car(
            env=self,
            car_path=os.path.join("assets", "sprites", "car.png"),
            block_size=100,
        )

        self.target = Target(
            env=self,
            target_path=os.path.join("assets", "sprites", "flag.png"),
        )

        self.obstacle = Obstacle(
            env=self,
            spawns=obstacles,
            block_size=100,
            image_path=os.path.join("assets", "sprites", "traffic-cone.png"),
        )

        self.displacement_to_target = helpers.distance_between_points(
            self.car.rect.center, self.target.rect.center
        )

    def draw_grid(self):
        blockSize = 100  # Set the size of the grid block
        white = (0, 0, 0)
        for x in range(0, self.screen.get_width(), blockSize):
            for y in range(0, self.screen.get_height(), blockSize):
                rect = pygame.Rect(x, y, blockSize, blockSize)
                pygame.draw.rect(self.screen, white, rect, 1)

    def step(self, action):
        self.truncation -= 1
        self.car.time_driving += 1
        reward = 0

        # previous_distance = helpers.distance_between_points(
        #     self.car.rect.center, self.target.rect.center
        # )

        if action == 0:
            result = self.car.move_left()
        elif action == 1:
            result = self.car.move_right()
        elif action == 2:
            result = self.car.move_up()
        elif action == 3:
            result = self.car.move_down()

        if not result:
            reward -= 10

        # Check collision with obstacles using the sprite group
        if pygame.sprite.spritecollide(self.car, self.obstacle.obstacles, False):
            self.done = True
            reward -= 100

        elif pygame.sprite.spritecollide(self.car, [self.target], False):
            self.done = True
            reward += 100

        else:
            reward -= 2

            if self.truncation < 0:
                reward -= 100
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
        car_x, car_y = self.map_to_grid(self.car.rect.center)
        target_x, target_y = self.map_to_grid(self.target.rect.center)
        obstacle_positions = [
            self.map_to_grid(obstacle.rect.center)
            for obstacle in self.obstacle.obstacles
        ]
        observation = np.zeros(shape=(8, 8), dtype=np.int8)

        # 1 - Car, 2 - Target, 3 - Obstacle
        observation[car_x][car_y] = 1
        observation[target_x][target_y] = 2

        for x, y in obstacle_positions:
            with contextlib.suppress(IndexError):
                # Moving obstacles can sometimes move out of bounds
                observation[x][y] = 3

        return observation

    def render(self):
        self.screen.fill((255, 255, 255))
        self.draw_grid()
        self.car.render()
        self.target.render()
        self.obstacle.render()
        pygame.display.flip()
        self.clock.tick(60)

    def reset(self, seed=None):
        self.done = False
        self.truncation = self.truncation_limit

        if seed:
            random.seed(seed)

        self.car.reset()
        self.target.reset()
        self.obstacle.reset()
        return (self._get_obs(), {})

    def close(self):
        pygame.quit()
        sys.exit()


if len(sys.argv) > 1:
    train = True
else:
    train = False

if train:
    env = CarEnv()
    check_env(env)
    env = DummyVecEnv([lambda: env])

    try:
        model = PPO.load("ppo_car", env=env)
    except FileNotFoundError:
        model = PPO("MlpPolicy", env, verbose=2, ent_coef=0.6)

    timesteps = int(sys.argv[1]) * 100_000
    model.learn(total_timesteps=timesteps, progress_bar=True)
    model.save("ppo_car")

    del model
    del env

env = CarEnv(obstacles=5)

try:
    model = PPO.load("ppo_car", env=env)
except FileNotFoundError:
    model = PPO("MlpPolicy", env, verbose=2)

# print(evaluate_policy(model, env, n_eval_episodes=100, deterministic=True))

done = False
score = 0
obs, info = env.reset()

import time

for i in range(500):
    while not done:
        env.render()

        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward

        time.sleep(0.5)

        if truncated or terminated:
            done = True

    print(score)
    score = 0
    done = False
    obs, info = env.reset()

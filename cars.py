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


class Car(pygame.sprite.Sprite):
    def __init__(
        self,
        screen,
        car_path=None,
        handling=5,
        max_speed=20,
        friction=0.1,
    ):
        super().__init__()
        self.screen = screen

        # Load and scale the car image
        self.original_image = pygame.image.load(car_path).convert_alpha()
        self.original_image = helpers.aspect_scale_image(self.original_image, 75)
        self.image = self.original_image.copy()

        # Set initial position and movement parameters
        self.position = (50, 50)
        self.speed = 0
        self.angle = 0
        self.move_angle = 0

        # Car properties
        self.default_handling = handling
        self.handling = handling
        self.max_speed = max_speed
        self.friction = friction

        # Control states
        self.accelerate = False
        self.reverse = False
        self.left = False
        self.right = False
        self.handbrake = False
        self.time_driving = 0

        # Car's rect (sprite's rect)
        self.rect = self.image.get_rect(center=self.position)

        # Store previous positions in memory (useful for collision or other checks)
        self.memory = []

    def get_reward(self):
        # fmt: off
        # Reward function based on car's position
        if (len(self.memory) > 1) and self.memory[-1] == self.memory[-2]:
            return -10
        # fmt: on
        return 0

    def update(self):
        # Update car's speed, movement, and fuel consumption
        if self.accelerate and self.speed < self.max_speed:
            self.speed += 0.5
        elif self.reverse and self.speed > -self.max_speed:
            self.speed -= 0.5
        elif self.speed > 0:
            self.speed -= self.friction
        elif self.speed < 0:
            self.speed += self.friction

        self.move_angle = self.angle

        # if not self.handbrake:
        #     self.move_angle = self.angle

        if self.speed > self.max_speed:
            self.speed = self.max_speed
        elif self.speed < -self.max_speed:
            self.speed = -self.max_speed

        if self.left:
            self.angle = (
                self.angle + (self.handling * (self.speed / self.max_speed))
            ) % 360
        if self.right:
            self.angle = (
                self.angle - (self.handling * (self.speed / self.max_speed))
            ) % 360

        if self.angle > self.move_angle:
            self.move_angle -= 1
        elif self.angle < self.move_angle:
            self.move_angle += 1
        else:
            self.move_angle = self.angle

        if self.speed < 0.05 and self.speed > -0.05:
            self.speed = 0

        if self.speed != 0:
            self.time_driving += 1
            new_x = self.rect.centerx + (
                self.speed * math.cos(math.radians(self.move_angle))
            )
            new_y = self.rect.centery - (
                self.speed * math.sin(math.radians(self.move_angle))
            )
            self.rect.center = (new_x, new_y)

        # Rotate the car image according to its angle
        self.image, self.rect = helpers.rotate_center(
            self.original_image, self.angle, self.rect.centerx, self.rect.centery
        )

        # Maintain memory length
        if len(self.memory) > 4:
            self.memory.pop(0)

    def render(self):
        # Render the car on the screen
        self.screen.blit(self.image, self.rect.topleft)
        # pygame.draw.rect(self.screen, (0, 255, 0), self.rect, 2)


class Target(pygame.sprite.Sprite):
    def __init__(
        self,
        screen,
        target_path=None,
    ):
        super().__init__()  # Initialize the pygame.sprite.Sprite
        self.screen = screen

        # Load and scale the target image
        self.image = pygame.image.load(target_path).convert_alpha()
        self.image = helpers.aspect_scale_image(self.image, 70)

        # Set the initial position
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
        screen,
        spawn_amount=5,
        obstacle_path=None,
    ):
        super().__init__()  # Initialize the pygame.sprite.Sprite
        self.screen = screen
        self.spawn_amount = spawn_amount
        self.obstacle_image = pygame.image.load(obstacle_path).convert_alpha()
        self.update()

    def _set_obstacle_position(self, obstacle_sprite):
        # Randomly position the obstacle, ensuring no overlap
        while True:
            position = (
                random.randint(75, self.screen.get_width() - 75),
                random.randint(75, self.screen.get_height() - 75),
            )
            obstacle_sprite.rect.center = position

            if not any(
                obstacle_sprite.rect.colliderect(other_obstacle.rect)
                for other_obstacle in self.obstacles
                if other_obstacle != obstacle_sprite
            ):
                break

    def get_corners(self, sprite):
        return [
            sprite.rect.topleft,
            sprite.rect.topright,
            sprite.rect.bottomleft,
            sprite.rect.bottomright,
        ]

    def render(self):
        # Draw all obstacles on the screen
        self.obstacles.draw(self.screen)

    def update(self):
        # Create a list to hold individual obstacle sprites
        self.obstacles = pygame.sprite.Group()

        # Create each obstacle and add it to the group
        for _ in range(self.spawn_amount):
            obstacle_sprite = pygame.sprite.Sprite()  # Create a new sprite
            obstacle_sprite.image = self.obstacle_image
            obstacle_sprite.image = helpers.aspect_scale_image(
                obstacle_sprite.image, 45
            )
            obstacle_sprite.rect = obstacle_sprite.image.get_rect()

            self._set_obstacle_position(obstacle_sprite)
            self.obstacles.add(obstacle_sprite)


class CarEnv(Env):
    def __init__(self, seed=None, render_mode="human"):
        screen_resolution = (800, 600)

        pygame.init()
        self.screen = pygame.display.set_mode(screen_resolution)
        self.clock = pygame.time.Clock()

        seed = 1
        if seed:
            random.seed(None)

        self.car = Car(
            screen=self.screen,
            car_path=os.path.join("assets", "sprites", "car.png"),
        )

        self.target = Target(
            screen=self.screen,
            target_path=os.path.join("assets", "sprites", "flag.png"),
        )

        self.displacement_to_target = helpers.distance_between_points(
            self.car.rect.center, self.target.rect.center
        )

        self.progress = 0  # 0%

        self.obstacle = Obstacle(
            screen=self.screen,
            obstacle_path=os.path.join("assets", "sprites", "traffic-cone.png"),
            spawn_amount=5,
        )

        self.truncation_limit = 1000
        self.truncation = self.truncation_limit

        # 0 - Left, 1 - None, 2 - Right
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.int8)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(44,), dtype=np.float32
        )

        self.done = False

    def draw_grid(self):
        blockSize = 100  # Set the size of the grid block
        white = (0, 0, 0)
        for x in range(0, self.screen.get_width(), blockSize):
            for y in range(0, self.screen.get_height(), blockSize):
                rect = pygame.Rect(x, y, blockSize, blockSize)
                pygame.draw.rect(self.screen, white, rect, 1)

    def step(self, action):
        self.truncation -= 1

        # Left or Right
        if action[0] == -1:
            self.car.re = True
            self.car.right = False
        elif action[0] == 0:
            self.car.right = False
            self.car.left = False
        elif action[0] == 1:
            self.car.right = True
            self.car.left = False

        if action[1] == -1:
            self.car.reverse = True
            self.car.handbrake = False
            self.car.accelerate = False
        elif action[1] == 0:
            self.car.reverse = False
            self.car.handbrake = True
            self.car.accelerate = False
        elif action[1] == 1:
            self.car.reverse = False
            self.car.handbrake = False
            self.car.accelerate = True

        self.car.update()
        reward = self.car.get_reward()

        # fmt: off
        # out of bounds
        if (
            (self.car.rect.centerx < -50) or (self.car.rect.centerx > self.screen.get_width() + 50)
            or (self.car.rect.centery < -50) or ( self.car.rect.centery > self.screen.get_height() + 50)
            ):
            self.done = True
            reward = -500
        # fmt: on

        # Check collision with obstacles using the sprite group
        if pygame.sprite.spritecollide(self.car, self.obstacle.obstacles, False):
            self.car.speed = 0
            self.done = True
            reward = -500

        elif pygame.sprite.spritecollide(self.car, [self.target], False):
            self.done = True
            reward += 500 + (self.truncation_limit - self.car.time_driving)

        else:
            # logic to reward based on distance to target
            distance_to_target = helpers.distance_between_points(
                self.car.rect.center, self.target.rect.center
            )

            distance_covered_percentage = (
                1 - (distance_to_target / self.displacement_to_target)
            ) * 100

            if distance_covered_percentage >= (self.progress):
                self.car.time_driving
                reward += 20
                self.progress += 10

            elif distance_covered_percentage < (self.progress - 10):
                reward -= 10
                self.progress -= 10

            # self.car.memory.append(distance_covered_percentage)

            # if len(self.car.memory) >= 2:
            #     # rewarding based on how much percentage of progress completed
            #     progress = self.car.memory[-1] - self.car.memory[-2]
            #     if progress > 0:
            #         reward += 1
            #     elif self.car.memory[-1] <= self.car.memory[-2]:
            #         reward -= 1

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
                "angle": self.car.angle,
                "speed": self.car.speed,
                "score": reward,
                "done": self.done,
            },
        )

    def _get_obs(self):
        obstacle_corners = np.array(
            [
                self.obstacle.get_corners(obstacle)
                for obstacle in self.obstacle.obstacles
            ],
            dtype=np.float32,
        ).flatten()

        return np.array(
            [
                *[self.car.rect.centerx, self.car.rect.centery],
                self.car.speed,
                self.car.angle,
                *obstacle_corners,
            ],
            dtype=np.float32,
        ).flatten()

    def render(self):
        self.screen.fill((255, 255, 255))
        self.draw_grid()
        self.all_sprites.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(60)

    def reset(self, seed=None):
        self.done = False
        self.truncation = self.truncation_limit

        seed = 1

        if seed:
            random.seed(seed)

        self.car = Car(
            screen=self.screen,
            car_path=os.path.join("assets", "sprites", "car.png"),
        )
        self.target = Target(
            screen=self.screen,
            target_path=os.path.join("assets", "sprites", "flag.png"),
        )

        self.obstacle = Obstacle(
            screen=self.screen,
            obstacle_path=os.path.join("assets", "sprites", "traffic-cone.png"),
            spawn_amount=5,
        )

        # Re-create the sprite group
        self.all_sprites = pygame.sprite.Group(
            self.car,
            self.target,
            *self.obstacle.obstacles,
        )

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
        model = PPO("MlpPolicy", env, verbose=2)

    timesteps = int(sys.argv[1]) * 100_000
    model.learn(total_timesteps=timesteps, progress_bar=True)
    model.save("ppo_car")

    del model
    del env

env = CarEnv()

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
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        score += reward

        if truncated or terminated:
            done = True

    print(score)
    score = 0
    done = False
    obs, info = env.reset()

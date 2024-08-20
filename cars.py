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
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import sys


class Car(pygame.sprite.Sprite):
    def __init__(
        self,
        screen,
        car_path=None,
        handling=5,
        max_speed=10,
        friction=0.1,
        mileage=80,
    ):
        super().__init__()
        self.screen = screen

        # Load and scale the car image
        self.original_image = pygame.image.load(car_path).convert_alpha()
        self.original_image = helpers.aspect_scale_image(self.original_image, 100)
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
        self.mileage = mileage

        # Control states
        self.accelerate = False
        self.reverse = False
        self.left = False
        self.right = False
        self.handbrake = False
        self.fuel_balance = mileage

        # Car's rect (sprite's rect)
        self.rect = self.image.get_rect(center=self.position)

        # Store previous positions in memory (useful for collision or other checks)
        self.memory = [self.rect.center]

    def get_reward(self):
        # Reward function based on car's position
        if self.rect.centerx < 0 or self.rect.centerx > self.screen.get_width():
            return -1
        if self.rect.centery < 0 or self.rect.centery > self.screen.get_height():
            return -1
        if (len(self.memory) > 1) and self.memory[-1] == self.memory[-2]:
            return -1
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

        if not self.handbrake:
            self.move_angle = self.angle

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
            self.fuel_balance -= 1 * abs(self.speed) / self.max_speed
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

        # Track movement in memory
        self.memory.append(self.rect.center)
        if len(self.memory) > 10:
            self.memory.pop(0)

    def render(self):
        # Render the car on the screen
        self.screen.blit(self.image, self.rect.topleft)


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
                random.randint(50, self.screen.get_width() - 50),
                random.randint(50, self.screen.get_height() - 50),
            )
            obstacle_sprite.rect.center = position

            if not any(
                obstacle_sprite.rect.colliderect(other_obstacle.rect)
                for other_obstacle in self.obstacles
                if other_obstacle != obstacle_sprite
            ):
                break

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
                obstacle_sprite.image, 65
            )
            obstacle_sprite.rect = obstacle_sprite.image.get_rect()

            self._set_obstacle_position(obstacle_sprite)
            self.obstacles.add(obstacle_sprite)


class CarEnv(Env):
    def __init__(self, seed=None, render_mode="human"):
        screen_width, screen_height = screen_resolution = (800, 600)

        pygame.init()
        self.screen = pygame.display.set_mode(screen_resolution)
        self.clock = pygame.time.Clock()

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

        # self.truncation_limit = 80 * 1
        self.truncation_limit = self.car.mileage
        self.truncation = self.truncation_limit

        # 0 - Left, 1 - None, 2 - Right
        self.action_space = Discrete(3)

        # fmt: off
        # Define the observation space
        obs_low = np.array([
            0, 0,                                               # car position (x, y)
            -360,                                               # car angle
            -self.car.max_speed,                                # car speed
            0,                                                  # fuel balance
            0,                                                  # distance
            0, 0,                                               # target position (x, y)
        ], dtype=np.float32)

        obs_high = np.array([
            screen_width, screen_height,                        # car position (x, y)
            360,                                                # car angle
            self.car.max_speed,                                 # car speed
            self.car.mileage,                                   # fuel balance 
            np.sqrt(screen_width**2 + screen_height**2),        # distance - maximum distance
            screen_width, screen_height,                        # target position (x, y)
        ], dtype=np.float32)
        # fmt: on

        self.observation_space = Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.done = False

    def step(self, action):
        self.truncation -= 1

        # Left or Right
        if action == 2:
            self.car.right = True
        elif action == 0:
            self.car.left = True
        else:
            self.car.right = False
            self.car.left = False

        self.car.accelerate = True
        self.car.handbrake = False

        self.car.update()

        # Check collision with obstacles using the sprite group
        if pygame.sprite.spritecollide(self.car, self.obstacle.obstacles, False):
            self.car.speed = 0
            self.done = True
            reward = -100
            return (
                self._get_obs(),
                reward,
                self.done,
                self.truncation <= 0,
                {
                    "fuel_balance": self.car.fuel_balance,
                    "position": self.car.rect.center,
                    "angle": self.car.angle,
                    "speed": self.car.speed,
                    "score": reward,
                    "done": self.done,
                },
            )

        reward = self.car.get_reward()

        distance_to_target = helpers.distance_between_points(
            self.car.rect.center, self.target.rect.center
        )

        previous_distance_to_target = helpers.distance_between_points(
            self.car.memory[-2], self.target.rect.center
        )

        if self.car.rect.colliderect(self.target.rect):
            self.car.fuel_balance = self.car.mileage
            self.truncation = self.truncation_limit
            reward += self.car.fuel_balance * 2
            reward += 100

            self.target.update()

        elif distance_to_target < previous_distance_to_target:
            reward += max(0, 1 - (distance_to_target / 1000))
        elif distance_to_target >= previous_distance_to_target:
            reward -= max(0, 1 - (distance_to_target / 1000))
        elif self.car.fuel_balance <= 0:
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
                "fuel_balance": self.car.fuel_balance,
                "position": self.car.rect.center,
                "angle": self.car.angle,
                "speed": self.car.speed,
                "score": reward,
                "done": self.done,
            },
        )

    def _get_obs(self):
        if self.car.fuel_balance < 0:
            self.car.fuel_balance = 0

        return np.array(
            [
                self.car.rect.centerx,
                self.car.rect.centery,
                self.car.angle,
                self.car.speed,
                self.car.fuel_balance,
                self.target.rect.centerx,
                self.target.rect.centery,
                helpers.distance_between_points(
                    self.car.rect.center, self.target.rect.center
                ),
            ],
            dtype=np.float32,
        )

    def render(self):
        self.screen.fill((255, 255, 255))
        self.all_sprites.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(60)

    def reset(self, seed=None):
        self.done = False
        self.truncation = self.truncation_limit

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
            self.car, self.target, *self.obstacle.obstacles
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

    num_envs = 4
    # env = SubprocVecEnv([env for _ in range(num_envs)])
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
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        score += reward

        # time.sleep(0.025)

        if truncated or terminated:
            done = True

    print(score)
    score = 0
    done = False
    obs, info = env.reset()

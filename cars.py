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
        stationary_spawns=5,
        moving_spawns=5,
        stationary_obstacle_path=None,
        moving_obstacle_path=None,
        block_size=100,
    ):
        super().__init__()  # Initialize the pygame.sprite.Sprite
        self.env = env
        self.screen = env.screen
        self.stationary_spawn_amount = stationary_spawns
        self.moving_spawn_amount = moving_spawns

        self.stationary_obstacle_image = pygame.image.load(
            stationary_obstacle_path
        ).convert_alpha()
        self.stationary_obstacle_image = helpers.aspect_scale_image(
            self.stationary_obstacle_image, 45
        )

        self.moving_obstacle_image = pygame.image.load(
            moving_obstacle_path
        ).convert_alpha()
        self.moving_obstacle_image = helpers.aspect_scale_image(
            self.moving_obstacle_image, 70
        )

        self.block_size = block_size
        self.update()

    def reset(self, seed=None):
        if seed:
            random.seed(seed)

        self.update()

    def _set_obstacle_position(self, obstacle_sprite):
        # Randomly position the obstacle, ensuring no overlap
        while True:
            x_position, y_position = helpers.generate_random_position(
                self.screen, self.block_size
            )

            if any(
                obstacle_sprite.rect.colliderect(other_obstacle.rect)
                for other_obstacle in self.obstacles
                if other_obstacle != obstacle_sprite
            ):
                continue
            elif x_position == y_position and x_position in [750, 50]:
                continue
            else:
                obstacle_sprite.rect.center = (x_position, y_position)
                break

        return obstacle_sprite

    def render(self):
        # Draw all obstacles on the screen
        self.obstacles.draw(self.screen)

    def update(self):
        if self.env.truncation == self.env.truncation_limit:
            # Initialize the obstacles group
            self.obstacles = pygame.sprite.Group()

            # Add stationary obstacles
            self.obstacles.add(
                self.create_stationary_obstacles()
                for _ in range(self.stationary_spawn_amount)
            )

            # Add moving obstacles
            self.obstacles.add(
                self.create_moving_obstacles() for _ in range(self.moving_spawn_amount)
            )

        else:
            for obstacle in self.obstacles:
                if obstacle.moving:
                    self.update_moving_obstacle(obstacle)

    def update_moving_obstacle(self, obstacle):
        rotate_flag = False
        # Move obstacle
        obstacle.rect.centerx, obstacle.rect.centery = helpers.transform_coordinates(
            position=(obstacle.rect.centerx, obstacle.rect.centery),
            angle=obstacle.angle,
            steps=self.block_size,
        )

        # Check for collisions with other obstacles
        if any(
            obstacle.rect.colliderect(other_obstacle.rect)
            for other_obstacle in self.obstacles
            if other_obstacle != obstacle
        ):
            rotate_flag = True

        # Check for out-of-bounds
        if (
            obstacle.rect.centerx < -50
            or obstacle.rect.centerx > self.screen.get_width() + 50
            or obstacle.rect.centery < -50
            or obstacle.rect.centery > self.screen.get_height() + 50
        ):
            rotate_flag = True

        if rotate_flag:
            # Reverse the move and adjust direction
            obstacle.rect.centerx, obstacle.rect.centery = (
                helpers.transform_coordinates(
                    position=(obstacle.rect.centerx, obstacle.rect.centery),
                    angle=obstacle.angle,
                    steps=-self.block_size,
                )
            )
            obstacle = self.set_direction(obstacle)
            obstacle.rect.centerx, obstacle.rect.centery = (
                helpers.transform_coordinates(
                    position=(obstacle.rect.centerx, obstacle.rect.centery),
                    angle=obstacle.angle,
                    steps=self.block_size,
                )
            )

    def create_stationary_obstacles(self):
        obstacle_sprite = pygame.sprite.Sprite()  # Create a new sprite
        obstacle_sprite.image = self.stationary_obstacle_image
        obstacle_sprite.rect = obstacle_sprite.image.get_rect()
        obstacle_sprite.moving = False
        return self._set_obstacle_position(obstacle_sprite)

    def create_moving_obstacles(self):
        obstacle_sprite = pygame.sprite.Sprite()
        obstacle_sprite.image = self.moving_obstacle_image
        obstacle_sprite.rect = obstacle_sprite.image.get_rect()
        obstacle_sprite.moving = True
        self.set_direction(obstacle_sprite)
        return self._set_obstacle_position(obstacle_sprite)

    def set_direction(self, obstacle):
        possible_angles = [0, 90, 180, 270]
        if hasattr(obstacle, "angle"):
            possible_angles.remove(obstacle.angle)
        obstacle.angle = random.choice(possible_angles)
        obstacle.image, obstacle.rect = helpers.rotate_center(
            self.moving_obstacle_image, obstacle.angle, obstacle.rect.center
        )
        return obstacle


class CarEnv(Env):
    def __init__(self, seed=None, render_mode="human"):
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
            low=0,
            high=3,
            shape=(self.screen.get_width() // 100, self.screen.get_height() // 100),
            dtype=np.float32,
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
            stationary_obstacle_path=os.path.join(
                "assets", "sprites", "traffic-cone.png"
            ),
            stationary_spawns=0,
            moving_obstacle_path=os.path.join("assets", "sprites", "car-alter.png"),
            moving_spawns=5,
            block_size=100,
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
        self.obstacle.update()
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

        # fmt: off
        # out of bounds
        if (
            (self.car.rect.centerx < -50) or (self.car.rect.centerx > self.screen.get_width() + 50)
            or (self.car.rect.centery < -50) or ( self.car.rect.centery > self.screen.get_height() + 50)
            ):
            self.done = True
            reward = -100
        # fmt: on

        # Check collision with obstacles using the sprite group
        if pygame.sprite.spritecollide(self.car, self.obstacle.obstacles, False):
            self.done = True
            reward -= 100

        elif pygame.sprite.spritecollide(self.car, [self.target], False):
            self.done = True
            reward += 100

        else:
            # # logic to reward based on distance to target
            # new_distance = helpers.distance_between_points(
            #     self.car.rect.center, self.target.rect.center
            # )

            # if previous_distance >= new_distance:
            #     reward -= 20

            # elif new_distance < previous_distance:
            #     reward += 40

            # for encouraging shorter path
            # reward -= 10
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
        observation = self.observation_space.low.copy()

        # 1 - Car, 2 - Target, 3 - Obstacle
        observation[car_x, car_y] = 1
        observation[target_x, target_y] = 2

        for x, y in obstacle_positions:
            with contextlib.suppress(IndexError):
                # Moving obstacles can sometimes move out of bounds
                observation[x, y] = 3

        observation = np.transpose(observation)
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
        model = PPO("MlpPolicy", env, verbose=2, ent_coef=0.3)

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
        env.render()

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward

        time.sleep(0.5)

        if truncated or terminated:
            done = True

    print(score)
    score = 0
    done = False
    obs, info = env.reset()

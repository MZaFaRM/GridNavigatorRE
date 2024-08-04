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


class Car:
    def __init__(
        self,
        screen,
        car_path=None,
        handling=7,
        max_speed=10,
        friction=0.1,
    ):
        self.original_car = pygame.image.load(car_path).convert_alpha()
        self.original_car = helpers.aspect_scale_image(self.original_car, 75)
        self.screen = screen
        self.car = self.original_car.copy()
        # self.position = (
        #     random.randint(50, self.screen.get_width() - 50),
        #     random.randint(50, self.screen.get_height() - 50),
        # )

        # print("car", self.position)

        self.position = (199, 285)

        self.car_rect = self.car.get_rect(center=self.position)
        pygame.draw.rect(self.screen, (255, 0, 0), self.car_rect, 2)  # debug

        self.speed = 0
        self.angle = 0
        self.move_angle = 0

        self.default_handling = handling
        self.handling = handling
        self.max_speed = max_speed
        self.friction = friction

        self.accelerate = False
        self.reverse = False
        self.left = False
        self.right = False
        self.handbrake = False
        self.fuel_balance = 200

        self.memory = [self.car_rect.center]

    def get_reward(self):
        if self.car_rect.centerx < 0 or self.car_rect.centerx > self.screen.get_width():
            return -1
        if (
            self.car_rect.centery < 0
            or self.car_rect.centery > self.screen.get_height()
        ):
            return -1
        if (len(self.memory) > 1) and self.memory[-1] == self.memory[-2]:
            return -1
        return 0

    def update(self):
        if self.accelerate and self.speed < self.max_speed:
            self.speed += 0.5
        elif self.reverse and self.speed > -self.max_speed:
            self.speed -= 0.5
        elif self.speed > 0:
            self.speed -= self.friction
        elif self.speed < 0:
            self.speed += self.friction

        if self.handbrake:
            if -0.05 < self.speed < 0.05:
                self.speed = 0
            elif self.speed > 0:
                self.speed -= 0.6
            elif self.speed < 0:
                self.speed += 0.6

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

        self.move_angle = self.angle

        # if not self.handbrake:
        #     self.move_angle = self.angle

        if self.speed < 0.05 and self.speed > -0.05:
            self.speed = 0

        if self.speed != 0:
            self.fuel_balance -= 1 * abs(self.speed) / self.max_speed
            new_x = self.car_rect.centerx + (
                self.speed * math.cos(math.radians(self.move_angle))
            )
            new_y = self.car_rect.centery - (
                self.speed * math.sin(math.radians(self.move_angle))
            )
            self.car_rect.center = (new_x, new_y)

        self.car, self.car_rect = helpers.rotate_center(
            self.original_car, self.angle, self.car_rect.centerx, self.car_rect.centery
        )

        self.memory.append(self.car_rect.center)

        if len(self.memory) > 10:
            self.memory.pop(0)

    def render(self):
        pygame.draw.rect(self.screen, (255, 0, 0), self.car_rect, 2)  # debug
        self.screen.blit(self.car, self.car_rect.topleft)


class Fuel:
    def __init__(
        self,
        screen,
        fuel_path=None,
    ):
        self.screen = screen

        self.fuel = pygame.image.load(fuel_path).convert_alpha()
        self.fuel = helpers.aspect_scale_image(self.fuel, 30)

        self.update()

    def render(self):
        pygame.draw.rect(self.screen, (255, 0, 0), self.fuel_rect, 2)  # debug
        self.screen.blit(self.fuel, self.position)

    def update(self):
        # self.position = (
        #     random.randint(50, self.screen.get_width() - 50),
        #     random.randint(50, self.screen.get_height() - 50),
        # )
        self.position = (515, 537)
        self.fuel_rect = self.fuel.get_rect(center=self.position)
        return self.position


class CarEnv(Env):
    def __init__(self, seed=None, render_mode="human"):
        screen_width, screen_height = screen_resolution = (800, 600)

        pygame.init()
        self.screen = pygame.display.set_mode(screen_resolution)
        self.clock = pygame.time.Clock()

        self.truncation_limit = 240
        self.truncation = self.truncation_limit

        if seed:
            random.seed(seed)

        self.car = Car(
            screen=self.screen,
            car_path=os.path.join("assets", "sprites", "car.png"),
        )

        self.fuel = Fuel(
            screen=self.screen,
            fuel_path=os.path.join("assets", "sprites", "fuel.png"),
        )

        # 0 - Left, 1 - None, 2 - Right
        self.action_space = Discrete(3)

        # fmt: off
        # Define the observation space
        obs_low = np.array([
            0, 0,                                               # car position (x, y)
            -360,                                               # car angle
            -self.car.max_speed,                                # car speed
            0,                                                  # fuel balance
            0, 0,                                               # fuel position (x, y)
            0                                                   # distance
        ], dtype=np.float32)

        obs_high = np.array([
            screen_width, screen_height,                        # car position (x, y)
            360,                                                # car angle
            self.car.max_speed,                                 # car speed - assuming a maximum speed
            200,                                                # fuel balance - assuming a maximum fuel
            screen_width, screen_height,                        # fuel position (x, y)
            np.sqrt(screen_width**2 + screen_height**2)         # distance - maximum distance
        ], dtype=np.float32)
        # fmt: on

        self.observation_space = Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.done = False

    def step(self, action):
        self.truncation -= 1

        # # Left or Right
        # if action[0] == 2:
        #     self.car.right = True
        # elif action[0] == 0:
        #     self.car.left = True
        # else:
        #     self.car.right = False
        #     self.car.left = False

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

        # # Accelerate or Reverse
        # if action[1] == 2:
        #     self.car.accelerate = True
        # elif action[1] == 0:
        #     self.car.reverse = True
        # else:
        #     self.car.accelerate = False
        #     self.car.reverse = False

        # # Handbrakes
        # if action[2] == 1:
        #     self.car.handling = 2 * self.car.default_handling
        #     self.car.handbrake = True
        # elif action[2] == 0:
        #     self.car.handbrake = False
        #     # if handbrake is pressed, double the handling
        #     self.car.handling = self.car.default_handling

        self.car.update()
        reward = self.car.get_reward()

        distance_to_fuel = helpers.distance_between_points(
            self.car.memory[-1], self.fuel.fuel_rect.center
        )

        previous_distance_to_fuel = helpers.distance_between_points(
            self.car.memory[-2], self.fuel.fuel_rect.center
        )

        if self.car.car_rect.colliderect(self.fuel.fuel_rect):
            self.done = True
            reward += self.car.fuel_balance
            reward += 100

            self.fuel.update()
            self.car.fuel_balance = 200

        elif distance_to_fuel < previous_distance_to_fuel:
            reward += max(0, 1 - (distance_to_fuel / 1000))
        elif distance_to_fuel >= previous_distance_to_fuel:
            reward -= max(0, 1 - (distance_to_fuel / 1000))
        elif self.car.fuel_balance <= 0:
            reward -= 100
            self.done = True

        observation = self._get_obs()

        return (
            observation,
            reward,
            self.done,
            self.truncation <= 0,
            {
                "fuel_balance": self.car.fuel_balance,
                "position": self.car.car_rect.center,
                "angle": self.car.angle,
                "speed": self.car.speed,
                "score": reward,
                "done": self.done,
            },
        )

    def _get_obs(self):
        # Get the current state as an image
        # return pygame.surfarray.array3d(pygame.display.get_surface())
        if self.car.fuel_balance < 0:
            self.car.fuel_balance = 0

        # fmt: off
        return np.array(
            [
                self.car.car_rect.centerx, self.car.car_rect.centery,
                self.car.angle,
                self.car.speed,
                self.car.fuel_balance,
                self.fuel.fuel_rect.centerx, self.fuel.fuel_rect.centery,
                helpers.distance_between_points(
                    self.car.car_rect.center, self.fuel.fuel_rect.center
                ),
            ],
            dtype=np.float32,
        )
        # fmt: on

    def render(self):
        self.screen.fill((255, 255, 255))
        self.car.render()
        self.fuel.render()
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
        self.fuel = Fuel(
            screen=self.screen,
            fuel_path=os.path.join("assets", "sprites", "fuel.png"),
        )
        return (self._get_obs(), {})

    def close(self):
        pygame.quit()
        sys.exit()


train = False
# train = False


if train:
    env = CarEnv(seed=100)
    check_env(env)

    num_envs = 4
    # env = SubprocVecEnv([env for _ in range(num_envs)])
    env = DummyVecEnv([lambda: env])

    try:
        model = PPO.load("ppo_car", env=env)
    except FileNotFoundError:
        model = PPO("MlpPolicy", env, verbose=2)

    model.learn(total_timesteps=500_000, progress_bar=True)
    model.save("ppo_car")

    del model
    del env

env = CarEnv(seed=100)

try:
    model = PPO.load("left_right_fixed", env=env)
except FileNotFoundError:
    model = PPO("MlpPolicy", env, verbose=2)

print(evaluate_policy(model, env, n_eval_episodes=100, deterministic=True))

done = False
score = 0
obs, info = env.reset(seed=100)

env.observation_space.sample()

for i in range(5):
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
    obs, info = env.reset(seed=100)

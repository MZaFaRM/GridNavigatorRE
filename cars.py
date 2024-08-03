import pygame
import os
import sys
import math
import helpers
import random
from gymnasium.spaces import Box


class Car:
    def __init__(
        self,
        screen_width,
        screen_height,
        screen,
        car_path=None,
        handling=3,
        max_speed=10,
        friction=0.1,
    ):
        self.original_car = pygame.image.load(car_path).convert_alpha()
        self.screen = screen

        self.screen_width = screen_width
        self.screen_height = screen_height

        self.original_car = helpers.aspect_scale_image(self.original_car, 75)
        self.car = self.original_car.copy()
        self.car_rect = self.car.get_rect(
            center=(screen_width // 2, screen_height // 2)
        )

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
        if self.car_rect.centerx < 0 or self.car_rect.centerx > self.screen_width:
            return -1
        if self.car_rect.centery < 0 or self.car_rect.centery > self.screen_height:
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
            self.speed *= 0.9

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

        if not self.handbrake:
            self.move_angle = self.angle

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

    def render(self):
        self.screen.blit(self.car, self.car_rect.topleft)


class Fuel:
    def __init__(self, screen_width, screen_height, screen, fuel_path=None):
        self.fuel = pygame.image.load(fuel_path).convert_alpha()
        self.screen = screen

        self.screen_width = screen_width
        self.screen_height = screen_height

        self.fuel = helpers.aspect_scale_image(self.fuel, 30)
        self.fuel_rect = self.fuel.get_rect(center=self.update())

    def render(self):
        self.screen.blit(self.fuel, self.position)

    def update(self):
        self.position = (
            random.randint(5, self.screen_width - 5),
            random.randint(5, self.screen_height - 5),
        )
        self.fuel_rect = self.fuel.get_rect(center=self.position)
        return self.position


class CarEnv:
    def __init__(self):
        pygame.init()
        screen_resolution = (800, 600)
        self.screen = pygame.display.set_mode(screen_resolution)
        self.clock = pygame.time.Clock()

        self.car = Car(
            screen_width=screen_resolution[0],
            screen_height=screen_resolution[1],
            screen=self.screen,
            car_path=os.path.join("assets", "sprites", "car.png"),
        )

        self.fuel = Fuel(
            screen_width=screen_resolution[0],
            screen_height=screen_resolution[1],
            screen=self.screen,
            fuel_path=os.path.join("assets", "sprites", "fuel.png"),
        )

        self.action_space = Box(-1, 1, shape=(3,), dtype=int)
        self.observation_space = Box(0, 255, shape=(*screen_resolution, 3), dtype=int)

        self.done = False

    def step(self, action):
        # Left or Right
        if action[0] == 1:
            self.car.right = True
        elif action[0] == -1:
            self.car.left = True
        else:
            self.car.right = False
            self.car.left = False

        # Accelerate or Reverse
        if action[1] == 1:
            self.car.accelerate = True
        elif action[1] == -1:
            self.car.reverse = True
        else:
            self.car.accelerate = False
            self.car.reverse = False

        # Handbrakes
        if action[2] == 1:
            self.car.handling *= 2
            self.car.handbrake = True
        else:
            self.car.handbrake = False
            # if handbrake is pressed, double the handling
            self.car.handling = self.car.default_handling

        self.car.update()
        reward = self.car.get_reward()

        if self.car.car_rect.colliderect(self.fuel.fuel_rect):
            self.fuel.update()
            self.car.fuel_balance = 200
            reward += 10
        elif helpers.distance_between_points(
            self.car.memory[-1], self.fuel.fuel_rect.center
        ) < helpers.distance_between_points(
            self.car.memory[-2], self.fuel.fuel_rect.center
        ):
            reward += 1
        elif self.car.fuel_balance <= 0:
            reward -= 100
            self.done = True

        observation = self._get_obs()

        return (
            observation,
            reward,
            self.done,
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
        return pygame.surfarray.array3d(pygame.display.get_surface())

    def render(self):
        self.screen.fill((255, 255, 255))
        self.car.render()
        self.fuel.render()
        pygame.display.flip()
        self.clock.tick(60)


env = CarEnv()
done = False

while not done:
    events = pygame.event.get() or [pygame.event.Event(pygame.NOEVENT)]

    for event in events:
        if event.type == pygame.QUIT:
            done = True
        else:
            action = helpers.translate_human_input(event)
            observation, reward, terminated, info = env.step(action)
            env.render()

            print(info)

            if terminated:
                done = True

pygame.quit()
sys.exit()

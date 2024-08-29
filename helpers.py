import random
import numpy as np
import pygame


def aspect_scale_image(image, width=None, height=None):
    original_width, original_height = image.get_size()
    if width and height:
        ratio = min(width / original_width, height / original_height)
    elif width:
        ratio = width / original_width
    elif height:
        ratio = height / original_height
    else:
        return image
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    return pygame.transform.scale(image, (new_width, new_height))


def rotate_center(image, angle, rect_center):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=rect_center)
    return rotated_image, new_rect


def distance_between_points(a, b):
    x1, y1 = a
    x2, y2 = b
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def translate_human_input(event, previous_action):
    action = previous_action.copy()

    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_d:
            action[0] = 2
        elif event.key == pygame.K_a:
            action[0] = 0
        elif event.key == pygame.K_w:
            action[1] = 2
        elif event.key == pygame.K_s:
            action[1] = 0
        elif event.key == pygame.K_LSHIFT:
            action[2] = 2

    elif event.type == pygame.KEYUP:
        if event.key == pygame.K_d:
            action[0] = 1
        elif event.key == pygame.K_a:
            action[0] = 1
        elif event.key == pygame.K_w:
            action[1] = 1
        elif event.key == pygame.K_s:
            action[1] = 1
        elif event.key == pygame.K_LSHIFT:
            action[2] = 1

    return action

def move_human_input(action, position, speed):
    x, y = position
    if action[0] == 0:
        x -= speed
    elif action[0] == 2:
        x += speed
    if action[1] == 0:
        y -= speed
    elif action[1] == 2:
        y += speed
    return x, y

def transform_coordinates(position, angle, steps):
    x, y = position
    x += steps * np.cos(np.radians(angle))
    y += steps * np.sin(np.radians(angle))
    return x, y


def generate_random_position(screen, block_size):
    x_position = (
        random.randint(1, ((screen.get_width() // block_size) - 1)) * 100
    ) - 50
    y_position = (
        random.randint(1, ((screen.get_height() // block_size) - 1)) * 100
    ) - 50

    return x_position, y_position

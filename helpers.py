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


def rotate_center(image, angle, x, y):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=(x, y))
    return rotated_image, new_rect


def distance_between_points(a, b):
    x1, y1 = a
    x2, y2 = b
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

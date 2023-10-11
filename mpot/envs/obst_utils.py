from math import ceil
import random
from mpot.envs.obst_map import ObstacleRectangle, ObstacleCircle


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return ceil(n * multiplier) / multiplier


def random_rect(xlim=(0, 0), ylim=(0, 0), width=2, height=2):
    """
    Generates an rectangular obstacle object, with random location and dimensions.
    """
    cx = random.uniform(xlim[0], xlim[1])
    cy = random.uniform(ylim[0], ylim[1])
    return ObstacleRectangle(cx, cy, width, height)


def random_circle(xlim=(0,0), ylim=(0,0), radius=2):
    """
    Generates a circle obstacle object, with random location and dimensions.
    """
    cx = random.uniform(xlim[0], xlim[1])
    cy = random.uniform(ylim[0], ylim[1])
    return ObstacleCircle(cx, cy, radius)

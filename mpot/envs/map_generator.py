import numpy as np
import torch

from mpot.envs.obst_map import ObstacleRectangle, ObstacleMap, ObstacleCircle
from mpot.envs.obst_utils import random_rect, random_circle
import copy

from torch_robotics.environments.primitives import MultiSphereField, ObjectField, MultiBoxField


def random_obstacles(
        map_dim = (1, 1),
        cell_size: float = 1.,
        num_obst: int = 5,
        rand_xy_limits=[[-1, 1], [-1, 1]],
        rand_rect_shape=[2, 2],
        rand_circle_radius: float = 1,
        max_attempts: int = 50,
        tensor_args=None,
):
    obst_map = ObstacleMap(map_dim, cell_size, tensor_args=tensor_args)
    num_boxes = np.random.randint(0, num_obst)
    num_circles = num_obst - num_boxes
    # randomize box obstacles
    xlim = rand_xy_limits[0]
    ylim = rand_xy_limits[1]
    width, height = rand_rect_shape

    boxes = []
    for _ in range(num_boxes):
        num_attempts = 0
        while num_attempts <= max_attempts:
            obst = random_rect(xlim, ylim, width, height)

            # Check validity of new obstacle
            # Do not overlap obstacles
            valid = obst._obstacle_collision_check(obst_map)
            if valid:
                # Add to Map
                obst._add_to_map(obst_map)
                # Add to list
                boxes.append(obst.to_array())
                break

            if num_attempts == max_attempts:
                print("Obstacle generation: Max. number of attempts reached. ")
                print(f"Total num. boxes: {len(boxes)}")
            num_attempts += 1
    boxes = torch.tensor(np.array(boxes), **tensor_args)
    cubes = MultiBoxField(boxes[:, :2], boxes[:, 2:], tensor_args=tensor_args)
    box_field = ObjectField([cubes], 'random-boxes')

    # randomize circle obstacles
    circles = []
    for _ in range(num_circles):
        num_attempts = 0
        while num_attempts <= max_attempts:
            obst = random_circle(xlim, ylim, rand_circle_radius)
            # Check validity of new obstacle
            # Do not overlap obstacles
            valid = obst._obstacle_collision_check(obst_map)

            if valid:
                # Add to Map
                obst._add_to_map(obst_map)
                # Add to list
                circles.append(obst.to_array())
                break

            if num_attempts == max_attempts:
                print("Obstacle generation: Max. number of attempts reached. ")
                print(f"Total num. boxes: {len(circles)}")

            num_attempts += 1
    circles = torch.tensor(np.array(circles), **tensor_args)
    spheres = MultiSphereField(circles[:, :2], circles[:, 2], tensor_args=tensor_args)
    sphere_field = ObjectField([spheres], 'random-spheres')
    obj_list = [box_field, sphere_field]
    obst_map.convert_map()
    return obst_map, obj_list


if __name__ == "__main__":
    cell_size = 0.1
    map_dim = [20, 20]
    seed = 2
    tensor_args = {'device': torch.device('cpu'), 'dtype': torch.float32}
    obst_map, obst_list = random_obstacles(
        map_dim, cell_size,
        num_obst=5,
        rand_xy_limits=[[-5, 5], [-5, 5]],
        rand_rect_shape=[2,2],
        rand_circle_radius=1,
        tensor_args=tensor_args
    )
    fig = obst_map.plot()

import numpy as np
import torch

from mpot.envs.obst_map import ObstacleRectangle, ObstacleMap, ObstacleCircle
from mpot.envs.obst_utils import random_rect, random_circle
import copy


def generate_obstacle_map(
        map_dim=(10, 10),
        obst_list=[],
        cell_size=1.,
        random_gen=False,
        num_obst=0,
        rand_xy_limits=None,
        rand_rect_shape=[2, 2],
        rand_circle_radius=1,
        tensor_args=None,
):

    """
    Args
    ---
    map_dim : (int,int)
        2D tuple containing dimensions of obstacle/occupancy grid.
        Treat as [x,y] coordinates. Origin is in the center.
        ** Dimensions must be an even number. **
    cell_sz : float
        size of each square map cell
    obst_list : [(cx_i, cy_i, width, height)]
        List of obstacle param tuples
    start_pts : float
        Array of x-y points for start configuration.
        Dim: [Num. of points, 2]
    goal_pts : float
        Array of x-y points for target configuration.
        Dim: [Num. of points, 2]
    seed : int or None
    random_gen : bool
        Specify whether to generate random obstacles. Will first generate obstacles provided by obst_list,
        then add random obstacles until number specified by num_obst.
    num_obst : int
        Total number of obstacles
    rand_limit: [[float, float],[float, float]]
        List defining x-y sampling bounds [[x_min, x_max], [y_min, y_max]]
    rand_shape: [float, float]
        Shape [width, height] of randomly generated obstacles.
    """
    ## Make occupancy grid
    obst_map = ObstacleMap(map_dim, cell_size, tensor_args=tensor_args)
    num_fixed = len(obst_list)
    for obst in obst_list:
        obst._add_to_map(obst_map)

    ## Add random obstacles
    obst_list = copy.deepcopy(obst_list)
    if random_gen:
        assert num_fixed <= num_obst, "Total number of obstacles must be greater than or equal to number specified in obst_list"
        xlim = rand_xy_limits[0]
        ylim = rand_xy_limits[1]
        width = rand_rect_shape[0]
        height = rand_rect_shape[1]
        radius = rand_circle_radius
        for _ in range(num_obst - num_fixed):
            num_attempts = 0
            max_attempts = 25
            while num_attempts <= max_attempts:
                if np.random.choice(2):
                    obst = random_rect(xlim, ylim, width, height)
                else:
                    obst = random_circle(xlim, ylim, radius)

                # Check validity of new obstacle
                # Do not overlap obstacles
                valid = obst._obstacle_collision_check(obst_map)

                if valid:
                    # Add to Map
                    obst._add_to_map(obst_map)
                    # Add to list
                    obst_list.append(obst)
                    break

                if num_attempts == max_attempts:
                    print("Obstacle generation: Max. number of attempts reached. ")
                    print("Total num. obstacles: {}.  Num. random obstacles: {}.\n"
                          .format( len(obst_list), len(obst_list) - num_fixed))

                num_attempts += 1

    obst_map.convert_map()
    return obst_map, obst_list


if __name__ == "__main__":
    import sys
    import numpy
    numpy.set_printoptions(threshold=sys.maxsize)
    obst_list = [
        ObstacleRectangle(0, 0, 2, 3),
        ObstacleCircle(-5, -5, 1)
    ]

    cell_size = 0.1
    map_dim = [20, 20]
    seed = 2
    tensor_args = {'device': torch.device('cpu'), 'dtype': torch.float32}
    obst_map, obst_list = generate_obstacle_map(
        map_dim, obst_list, cell_size,
        random_gen=True,
        # random_gen=False,
        num_obst=5,
        rand_xy_limits=[[-5, 5], [-5, 5]],
        rand_rect_shape=[2,2],
        rand_circle_radius=1,
        tensor_args=tensor_args
    )

    fig = obst_map.plot()

    traj_y = torch.linspace(-map_dim[1]/2., map_dim[1]/2., 20)
    traj_x = torch.zeros_like(traj_y)
    X = torch.cat((traj_x.unsqueeze(1), traj_y.unsqueeze(1)), dim=1)
    cost = obst_map.get_collisions(X)
    print(cost)

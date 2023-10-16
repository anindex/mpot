import torch
from matplotlib import pyplot as plt

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.robots import RobotPointMass
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes
from mpot.envs.map_generator import random_obstacles


class EnvOccupancy2D(EnvBase):

    def __init__(self, tensor_args=None, **kwargs):
        if tensor_args is None:
            tensor_args = DEFAULT_TENSOR_ARGS
        obst_map, obj_list = random_obstacles(
            map_dim=[20, 20],
            cell_size=0.1,
            num_obst=15,
            rand_xy_limits=[[-7.5, 7.5], [-7.5, 7.5]],
            rand_rect_shape=[2, 2],
            rand_circle_radius=1.,
            tensor_args=tensor_args
        )

        super().__init__(
            name=self.__class__.__name__,
            limits=torch.tensor([[-10, -10], [10, 10]], **tensor_args),  # environments limits
            obj_fixed_list=obj_list,
            tensor_args=tensor_args,
            **kwargs
        )
        self.occupancy_map = obst_map


if __name__ == '__main__':
    env = EnvOccupancy2D(precompute_sdf_obj_fixed=False, tensor_args=DEFAULT_TENSOR_ARGS)
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()

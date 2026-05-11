import os
from pathlib import Path
import time
import matplotlib.pyplot as plt
import torch

from mpot.ot.problem import EpsilonScheduler
from mpot.ot.sinkhorn import Sinkhorn
from mpot.planner import MPOT
from mpot.costs import CostGPHolonomic, CostField, CostComposite
from mpot.envs.occupancy import EnvOccupancy2D
from mpot.utils.trajectory import interpolate_trajectory

from torch_robotics.robots.robot_point_mass import RobotPointMass
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer


if __name__ == "__main__":
    seed = int(time.time())
    fix_random_seed(seed)

    device = get_torch_device()
    tensor_args = {'device': device, 'dtype': torch.float32}

    # ---------------------------- Environment, Robot, PlanningTask ---------------------------------
    q_limits = torch.tensor([[-10, -10], [10, 10]], **tensor_args)
    env = EnvOccupancy2D(
        precompute_sdf_obj_fixed=False,
        tensor_args=tensor_args
    )

    robot = RobotPointMass(
        q_limits=q_limits,
        tensor_args=tensor_args
    )

    task = PlanningTask(
        env=env,
        robot=robot,
        ws_limits=q_limits,
        obstacle_cutoff_margin=0.05,
        tensor_args=tensor_args
    )

    # -------------------------------- Params ---------------------------------
    step_radius = 0.15
    probe_radius = 0.15

    polytope = 'cube'  # 'simplex' | 'orthoplex' | 'cube'

    epsilon_sched = EpsilonScheduler(target=1e-2, init=1.0, decay=1.0)
    ent_epsilon = EpsilonScheduler(target=1e-2, init=1.0, decay=1.0)

    num_probe = 5
    num_particles_per_goal = 33
    pos_limits = [-10, 10]
    vel_limits = [-10, 10]
    w_coll = 5e-3
    w_smooth = 1e-7
    sigma_gp = 0.1
    sigma_gp_init = 1.6
    max_inner_iters = 100
    max_outer_iters = 100

    start_state = torch.tensor([-9, -9, 0., 0.], **tensor_args)

    multi_goal_states = torch.tensor([
        [0, 9, 0., 0.],
        [9, 9, 0., 0.],
        [9, 0, 0., 0.]
    ], **tensor_args)

    traj_len = 64
    dt = 0.1

    # --------------------------------- Cost function ---------------------------------
    cost_coll = CostField(
        robot, traj_len,
        field=env.occupancy_map,
        sigma=1.0,
        tensor_args=tensor_args
    )
    cost_gp = CostGPHolonomic(robot, traj_len, dt, sigma_gp, [0, 1], tensor_args=tensor_args)
    cost_func_list = [cost_coll, cost_gp]
    weights_cost_l = [w_coll, w_smooth]
    cost = CostComposite(
        robot, traj_len, cost_func_list,
        weights_cost_l=weights_cost_l,
        tensor_args=tensor_args
    )

    # --------------------------------- MPOT Init ---------------------------------

    linear_ot_solver = Sinkhorn(
        threshold=1e-6,
        inner_iterations=1,
        max_iterations=max_inner_iters,
    )

    ss_params = dict(
        epsilon=epsilon_sched,
        ent_epsilon=ent_epsilon,
        step_radius=step_radius,
        probe_radius=probe_radius,
        num_probe=num_probe,
        min_iterations=5,
        max_iterations=max_outer_iters,
        threshold=2e-3,
        store_history=True,
        tensor_args=tensor_args,
    )

    mpot_params = dict(
        objective_fn=cost,
        linear_ot_solver=linear_ot_solver,
        ss_params=ss_params,
        dim=2,
        traj_len=traj_len,
        num_particles_per_goal=num_particles_per_goal,
        dt=dt,
        start_state=start_state,
        multi_goal_states=multi_goal_states,
        pos_limits=pos_limits,
        vel_limits=vel_limits,
        polytope=polytope,
        fixed_goal=True,
        sigma_start_init=0.001,
        sigma_goal_init=0.001,
        sigma_gp_init=sigma_gp_init,
        seed=seed,
        tensor_args=tensor_args,
    )
    planner = MPOT(**mpot_params)

    # --------------------------------- Optimize ---------------------------------

    with TimerCUDA() as t:
        trajs, optim_state, opt_iters = planner.optimize()
    int_trajs = interpolate_trajectory(trajs, num_interpolation=3)
    colls = env.occupancy_map.get_collisions(int_trajs[..., :2]).any(dim=1)
    sinkhorn_iters = optim_state.linear_convergence[:opt_iters]
    print(f'Optimization finished at {opt_iters}! Parallelization Quality (GOOD [%]): {(1 - colls.float().mean()) * 100:.2f}')
    print(f'Time(s) optim: {t.elapsed} sec')
    print(f'Average Sinkhorn Iterations: {sinkhorn_iters.mean():.2f}, min: {sinkhorn_iters.min():.2f}, max: {sinkhorn_iters.max():.2f}')

    # -------------------------------- Visualize ---------------------------------
    planner_visualizer = PlanningVisualizer(
        task=task,
        planner=planner
    )

    traj_history = optim_state.X_history[:opt_iters]
    traj_history = traj_history.view(opt_iters, -1, traj_len, 4)
    base_file_name = Path(os.path.basename(__file__)).stem
    pos_trajs_iters = robot.get_position(traj_history)

    planner_visualizer.animate_opt_iters_joint_space_state(
        trajs=traj_history,
        pos_start_state=start_state,
        vel_start_state=torch.zeros_like(start_state),
        video_filepath=f'{base_file_name}-joint-space-opt-iters.mp4',
        n_frames=max((2, opt_iters // 5)),
        anim_time=5
    )

    planner_visualizer.animate_opt_iters_robots(
        trajs=pos_trajs_iters, start_state=start_state,
        video_filepath=f'{base_file_name}-traj-opt-iters.mp4',
        n_frames=max((2, opt_iters // 5)),
        anim_time=5
    )

    plt.show()

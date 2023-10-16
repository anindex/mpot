
import os
from pathlib import Path
import time
import matplotlib.pyplot as plt
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from mpot.ot.problem import Epsilon
from mpot.ot.sinkhorn import Sinkhorn
from mpot.planner import MPOT
from mpot.costs import CostGPHolonomic, CostField, CostComposite

from torch_robotics.environments.env_spheres_3d import EnvSpheres3D
from torch_robotics.robots.robot_panda import RobotPanda
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer

allow_ops_in_compiled_graph()


if __name__ == "__main__":
    base_file_name = Path(os.path.basename(__file__)).stem

    seed = int(time.time())
    fix_random_seed(seed)

    device = get_torch_device()
    tensor_args = {'device': device, 'dtype': torch.float32}

    # ---------------------------- Environment, Robot, PlanningTask ---------------------------------
    env = EnvSpheres3D(
        precompute_sdf_obj_fixed=False,
        sdf_cell_size=0.01,
        tensor_args=tensor_args
    )

    robot = RobotPanda(
        use_self_collision_storm=False,
        tensor_args=tensor_args
    )

    task = PlanningTask(
        env=env,
        robot=robot,
        ws_limits=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], **tensor_args),  # workspace limits
        obstacle_cutoff_margin=0.03,
        tensor_args=tensor_args
    )

    # -------------------------------- Params ---------------------------------
    for _ in range(100):
        q_free = task.random_coll_free_q(n_samples=2)
        start_state = q_free[0]
        goal_state = q_free[1]

        # check if the EE positions are "enough" far apart
        start_state_ee_pos = robot.get_EE_position(start_state).squeeze()
        goal_state_ee_pos = robot.get_EE_position(goal_state).squeeze()

        if torch.linalg.norm(start_state_ee_pos - goal_state_ee_pos) > 0.5:
            break

    # start_state = torch.tensor([1.0403,  0.0493,  0.0251, -1.2673,  1.6676,  3.3611, -1.5428], **tensor_args)
    # goal_state = torch.tensor([1.1142,  1.7289, -0.1771, -0.9284,  2.7171,  1.2497,  1.7724], **tensor_args)

    print('Start state: ', start_state)
    print('Goal state: ', goal_state)
    start_state = torch.concatenate((start_state, torch.zeros_like(start_state)))
    goal_state = torch.concatenate((goal_state, torch.zeros_like(goal_state)))
    multi_goal_states = goal_state.unsqueeze(0)

    # Construct planner
    duration = 5  # sec
    traj_len = 64
    dt = duration / traj_len
    num_particles_per_goal = 10  # number of plans per goal  # NOTE: if memory is not enough, reduce this number

    # NOTE: these parameters are tuned for this environment
    step_radius = 0.03
    probe_radius = 0.08  # probe radius >= step radius

    # NOTE: changing polytope may require tuning again 
    # NOTE: cube in this case could lead to memory insufficiency, depending how many plans are optimized
    polytope = 'cube'  # 'simplex' | 'orthoplex' | 'cube';

    epsilon = 0.02
    ent_epsilon = Epsilon(1e-2)
    num_probe = 3  # number of probes points for each polytope vertices
    # panda joint limits
    q_max = torch.tensor([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159], **tensor_args)
    q_min = torch.tensor([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159], **tensor_args)
    pos_limits = torch.stack([q_min, q_max], dim=1)
    vel_limits = [-5, 5]
    w_coll = 1e-1  # for tuning the obstacle cost
    w_smooth = 1e-7  # for tuning the GP cost: error = w_smooth * || Phi x(t) - x(1+1) ||^2
    sigma_gp = 0.01  # for tuning the GP cost: Q_c = sigma_gp^2 * I
    sigma_gp_init = 0.5   # for controlling the initial GP variance: Q0_c = sigma_gp_init^2 * I
    max_inner_iters = 100  # max inner iterations for Sinkhorn-Knopp
    max_outer_iters = 40  # max outer iterations for MPOT

    #--------------------------------- Cost function ---------------------------------

    cost_func_list = []
    weights_cost_l = []
    for collision_field in task.get_collision_fields():
        cost_func_list.append(
            CostField(
                robot, traj_len,
                field=collision_field,
                sigma_coll=1.0,
                tensor_args=tensor_args
            )
        )
        weights_cost_l.append(w_coll)
    cost_gp = CostGPHolonomic(robot, traj_len, dt, sigma_gp, [0, 1], weight=w_smooth, tensor_args=tensor_args)
    cost_func_list.append(cost_gp)
    weights_cost_l.append(w_smooth)
    cost = CostComposite(
        robot, traj_len, cost_func_list,
        weights_cost_l=weights_cost_l,
        tensor_args=tensor_args
    )

    #--------------------------------- MPOT Init ---------------------------------

    linear_ot_solver = Sinkhorn(
        threshold=1e-3,
        inner_iterations=1,
        max_iterations=max_inner_iters,
    )
    ss_params = dict(
        epsilon=epsilon,
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
        dim=7,
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

    # Optimize
    with TimerCUDA() as t:
        trajs, optim_state, opt_iters = planner.optimize()
    sinkhorn_iters = optim_state.linear_convergence[:opt_iters]
    print(f'Optimization finished at {opt_iters}! Optimization time: {t.elapsed:.3f} sec')
    print(f'Average Sinkhorn Iterations: {sinkhorn_iters.mean():.2f}, min: {sinkhorn_iters.min():.2f}, max: {sinkhorn_iters.max():.2f}')

    # -------------------------------- Visualize ---------------------------------
    planner_visualizer = PlanningVisualizer(
        task=task,
        planner=planner
    )

    traj_history = optim_state.X_history[:opt_iters]
    traj_history = traj_history.view(opt_iters, -1, traj_len, 14)  # 7 + 7
    pos_trajs_iters = robot.get_position(traj_history)
    trajs = trajs.flatten(0, 1)
    trajs_coll, trajs_free = task.get_trajs_collision_and_free(trajs)

    planner_visualizer.animate_opt_iters_joint_space_state(
        trajs=traj_history,
        pos_start_state=start_state, pos_goal_state=goal_state,
        vel_start_state=torch.zeros_like(start_state), vel_goal_state=torch.zeros_like(goal_state),
        video_filepath=f'{base_file_name}-joint-space-opt-iters.mp4',
        n_frames=max((2, opt_iters // 2)),
        anim_time=5
    )

    if trajs_free is not None:
        planner_visualizer.animate_robot_trajectories(
            trajs=trajs_free, start_state=start_state, goal_state=goal_state,
            plot_trajs=False,
            draw_links_spheres=False,
            video_filepath=f'{base_file_name}-robot-traj.mp4',
            # n_frames=max((2, pos_trajs_iters[-1].shape[1]//10)),
            n_frames=trajs_free.shape[-2],
            anim_time=duration
        )

    plt.show()

import torch
import time
import random
import matplotlib.pyplot as plt
from matplotlib import animation, rc
rc('animation', html='html5')
save_path = 'data/planar'
import numpy as np
import os

from mpot.envs.map_generator import generate_obstacle_map
from mpot.ot.problem import Epsilon, LinearEpsilon
from mpot.ot.sinkhorn import Sinkhorn
from mpot.planner import MPOT
from mpot.costs import CostComposite, CostField, CostGPHolonomic

from torch_robotics.torch_utils.torch_timer import TimerCUDA


def plot_history(i):
    X = X_hist[i]
    X = X.view(3 * num_particles_per_goal, -1, 4)
    # free trajs flag
    colls = obst_map.get_collisions(X[:, :, :2]).any(dim=1)
    X = X.cpu().numpy()
    
    for j in range(X.shape[0]):
        if colls[j]:
            points[j].set_color('black')
            lines[j].set_color('black')
        else:
            points[j].set_color('red')
            lines[j].set_color('red')
        points[j].set_data(X[j, :, 0], X[j, :, 1])
        lines[j].set_data(X[j, :, 0], X[j, :, 1])
    text.set_text(f'Iters: {i}')
    return *points, *lines


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print('CUDA is not available. Using CPU instead!')
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    tensor_args = {'device': device, 'dtype': torch.float32}
    seed = int(time.time())
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    dim = 2  # state_dim = 4 for velocity
    dt = 0.1
    traj_len = 64
    view_optim = True   # plot optimization progress

    #--------------------------------- MPOT Tuning Parameters -------------------------------------
    # NOTE: these parameters are tuned for the planar environment
    step_radius = 0.35
    probe_radius = 0.5  # probe radius > step radius

    # NOTE: changing polytope may require tuning again
    polytope = 'cube'  # 'random' | 'simplex' | 'orthoplex' | 'cube'; 'random' option is added for ablations, not recommended for general use

    epsilon = 0.02
    num_probe = 5  # number of probes points for each polytope vertices
    num_particles_per_goal = 33  # number of plans per goal
    pos_limits = [-10, 10]
    vel_limits = [-10, 10]
    w_coll = 2.4e-3  # for tuning the obstacle cost
    w_smooth = 1e-7  # for tuning the GP cost: error = w_smooth * || Phi x(t) - x(1+1) ||^2
    sigma_gp = 0.13   # for tuning the GP cost: Q_c = sigma_gp^2 * I
    sigma_gp_init = 1.5   # for controlling the initial GP variance: Q0_c = sigma_gp_init^2 * I
    max_inner_iters = 100  # max inner iterations for Sinkhorn-Knopp
    max_outer_iters = 100  # max outer iterations for MPOT
    #--------------------------------- Task Settings ----------------------------------------------

    start_state = torch.tensor([-9, -9], **tensor_args)

    # NOTE: change goal states here
    multi_goal_states = torch.tensor([
        [0, 9],
        [9, 9],
        [9, 0]
    ], **tensor_args)

    ## Obstacle map
    obst_params = dict(
        map_dim=[20, 20],
        obst_list=[],
        cell_size=0.1,
        map_type='direct',
        random_gen=True,
        num_obst=15,
        rand_xy_limits=[[-7.5, 7.5], [-7.5, 7.5]],
        tensor_args=tensor_args,
    )
    # For obst. generation
    random.seed(seed)
    np.random.seed(seed)
    obst_map = generate_obstacle_map(**obst_params)[0]

    #--------------------------------- Cost functions ---------------------------------

    # Construct cost function
    cost_gp = CostGPHolonomic(dim, traj_len, dt, sigma_gp, [0, 1], weight=w_smooth, tensor_args=tensor_args)
    cost_obst_2D = CostField(dim, [0, traj_len], field=obst_map, weight=w_coll, tensor_args=tensor_args)
    cost_func_list = [cost_obst_2D, cost_gp]
    objective_fn = CostComposite(dim, cost_func_list, tensor_args=tensor_args)

    #--------------------------------- MPOT Init ---------------------------------
    # Sinkhorn-Knopp parameters
    linear_ot_solver = Sinkhorn(
        threshold=1e-3,
        max_iterations=max_inner_iters
    )
    ent_epsilon = Epsilon(0.05)
    ss_params = dict(
        epsilon=epsilon,
        ent_epsilon=ent_epsilon,
        step_radius=step_radius,
        probe_radius=probe_radius,
        num_probe=num_probe,
        min_iterations=5,
        max_iterations=max_outer_iters,
        threshold=2e-4,
        store_history=True,
        tensor_args=tensor_args,
    )

    mpot_params = dict(
        objective_fn=objective_fn,
        linear_ot_solver=linear_ot_solver,
        ss_params=ss_params,
        traj_len=traj_len,
        num_particles_per_goal=num_particles_per_goal,
        dt=dt,
        start_state=start_state,
        multi_goal_states=multi_goal_states,
        pos_limits=pos_limits,
        vel_limits=vel_limits,
        polytope=polytope,
        sigma_start_init=0.001,
        sigma_goal_init=0.001,
        sigma_gp_init=sigma_gp_init,
        seed=seed,
        tensor_args=tensor_args,
    )
    mpot = MPOT(**mpot_params)

    #--------------------------------- Optimize ---------------------------------

    with TimerCUDA() as t:
        trajs, optim_state, iter = mpot.optimize()
    colls = obst_map.get_collisions(trajs[..., :2]).any(dim=1)
    print(f'Optimization finished at {iter}! Parallelization Quality (GOOD [%]): {(1 - colls.float().mean()) * 100:.2f}')
    print(f'Time(s) optim: {t.elapsed} sec')
    if view_optim:
        os.makedirs(save_path, exist_ok=True)
        X_hist = optim_state.X_history[:iter]
        fig, ax = plt.subplots()
        points = []
        lines = []
        for i in range(3 * num_particles_per_goal):
            point, = ax.plot([], [], 'ro', alpha=0.7, markersize=3)
            line,  = ax.plot([], [], 'r-', alpha=0.5, linewidth=0.5)
            points.append(point)
            lines.append(line)
        text = ax.text(10, 11, f'Iters {i}', style='italic')
        ax.set_xlim((-10, 10))
        ax.set_ylim((-10, 10))
        fig.tight_layout()
        ax.axis('off')
        ax.set_aspect('equal')
        cs = ax.contourf(x, y, obst_map.map, 20, cmap='Greys', alpha=1)
        multi_goal_states = multi_goal_states.cpu().numpy()
        for i in range(multi_goal_states.shape[0]):
            ax.plot(multi_goal_states[i, 0], multi_goal_states[i, 1], 'go', markersize=5)
        fig.tight_layout()

        print('Saving animation..., please wait')
        anim = animation.FuncAnimation(fig, plot_history, frames=len(X_hist), interval=20, blit=True)
        anim.save('data/planar/planar.gif', writer='imagemagick', fps=30)

    #--------------------------------- Plotting ---------------------------------  

    fig = plt.figure()
    ax = fig.gca()
    cs = ax.contourf(x, y, obst_map.map, 20, cmap='Greys', alpha=1.)
    ax.set_aspect('equal')
    trajs = trajs.cpu().numpy()
    vel_trajs = trajs[:, :, :, 2:]
    for i in range(trajs.shape[0]):
        for j in range(trajs.shape[1]):
            # ax.plot(trajs[i, j, :, 0], trajs[i, j, :, 1], 'bo', markersize=3)
            ax.plot(trajs[i, j, :, 0], trajs[i, j, :, 1], 'b-', alpha=0.5, linewidth=0.5)
    # plot goal
    for i in range(multi_goal_states.shape[0]):
        ax.plot(multi_goal_states[i, 0], multi_goal_states[i, 1], 'go', markersize=5)
    ax.axis('off')
    fig.tight_layout()
    plt.show()

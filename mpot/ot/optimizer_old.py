import torch

from mpot.utils.polytopes import POLYTOPE_MAP, get_sampled_points_on_sphere, get_sampled_polytope_vertices
from mpot.utils.probe import get_projecting_points, get_shifted_points
from mpot.ot.sinkhorn_old import sinkhorn_knopp_stabilized


def scale_cost_matrix(M):
    min_M = M.min()
    if min_M < 0:
        M -= min_M
    max_M = M.max()
    if max_M > 1.:
        M /= max_M   # for stability
    return M


def update_inner_log_dict(log_dict, inner_log_dict):
    log_dict['err'].append(inner_log_dict['err'])
    log_dict['inner_iter'].append(inner_log_dict['inner_iter'])
    log_dict['logu'].append(inner_log_dict['logu'])
    log_dict['logv'].append(inner_log_dict['logv'])
    log_dict['a'].append(inner_log_dict['a'])
    log_dict['b'].append(inner_log_dict['b'])


def sinkhorn_step(step_dist, step_weights, X, cost, step_radius, probe_radius, beta=None, reg=0.1, numItermax=1000, verbose=False, log=False, polytope_vertices=None, **kwargs):
    polytope = kwargs.get('polytope', 'orthoplex')
    num_probe = kwargs.get('num_probe', 5)
    probe_step_size = kwargs.get('probe_step_size', 0.1)
    pos_scaler = kwargs.get('pos_scaler', None)
    vel_scaler = kwargs.get('vel_scaler', None)
    innerStopThr = kwargs.pop('innerStopThr', 1.e-9)
    num_sphere_point = kwargs.get('num_sphere_point', X.shape[0])
    k = X.shape[0]
    if beta is None:
        beta = torch.full(k, 1 / k).type_as(M)
    # scale features
    X_scaled = X.clone()
    if pos_scaler is not None and vel_scaler is not None:
        pos_scaler(X_scaled)
        vel_scaler(X_scaled)

    if step_dist is None:
        if polytope != 'random':
            step_points, probe_points, vertices = get_sampled_polytope_vertices(X_scaled, polytope_vertices=polytope_vertices, step_radius=step_radius, probe_radius=probe_radius, 
                                                                                num_probe=num_probe, random_probe=False, num_sphere_point=num_sphere_point)
        else:
            step_points, probe_points, vertices = get_sampled_points_on_sphere(X_scaled, step_radius=step_radius, probe_radius=probe_radius, num_probe=num_probe, 
                                                                               random_probe=False, num_sphere_point=num_sphere_point)
        if step_weights is None:
            if polytope != 'random':
                step_weights = torch.full((vertices.shape[0], ), 1 / vertices.shape[0]).type_as(X)
            else:
                step_weights = torch.full((num_sphere_point, ), 1 / num_sphere_point).type_as(X)
                
        if pos_scaler is not None and vel_scaler is not None:
            pos_scaler.inverse(step_points)
            vel_scaler.inverse(step_points)
            pos_scaler.inverse(probe_points)
            vel_scaler.inverse(probe_points)
    else:
        step_points = step_dist
        step_points = get_shifted_points(X_scaled, step_points)
        if pos_scaler is not None and vel_scaler is not None:
            pos_scaler.inverse(step_points)
            vel_scaler.inverse(step_points)

        probe_points = get_projecting_points(X, step_points, probe_step_size, num_probe)

    # compute cost matrix
    M = cost.eval(probe_points, current_trajs=X, **kwargs)
    M = scale_cost_matrix(M)  # for numerical stability
    # perform sinkhorn-knopp
    T, log_dict = sinkhorn_knopp_stabilized(beta, step_weights, M, reg, numItermax=numItermax, stopThr=innerStopThr, verbose=verbose, log=False, **kwargs)
    # perform stepping
    X_new = torch.einsum('ij,ijk->ik', T * k, step_points)
    return X_new, log_dict


def optimize(step_dist, step_weights, X_init, cost, step_radius, probe_radius, reg=0.01, numItermax=100, numInnerItermax=1000, stopThr=1e-7, verbose=False, log=False, **kwargs):
    update_start = kwargs.get('update_start', False)
    update_end = kwargs.get('update_end', False)
    annealing = kwargs.get('annealing', False)
    eps_annealing = kwargs.get('eps_annealing', 0.1)
    polytope = kwargs.get('polytope', 'orthoplex')
    traj_dim = kwargs.get('traj_dim', None)  # [num_goal, num_particle_per_goal, traj_len, n_dof] or [traj_len, n_dof]

    iter_count = 0
    k, d = X_init.shape[0], X_init.shape[1]
    polytope_vertices = None
    if step_dist is None and polytope != 'random':
        polytope_vertices = POLYTOPE_MAP[polytope](torch.zeros(d).type_as(X_init))
        step_weights = torch.full((polytope_vertices.shape[0], ), 1 / polytope_vertices.shape[0]).type_as(X_init)
    beta = torch.full((k, ), 1 / k).type_as(X_init)
    X = X_init
    log_dict = {
        'traj_cost': [], 'err': [], 'inner_iter': [],
        'logu': [], 'logv': [],
        'a': [], 'b': [],
    }
    displacement_square_norms = []
    X_hist = []
    displacement_square_norm = stopThr + 1.

    while (displacement_square_norm > stopThr and iter_count < numItermax):
        # perform the sinkhorn steps
        T_sum, inner_log_dict = sinkhorn_step(step_dist, step_weights, X, cost, step_radius, probe_radius, beta=beta, reg=reg, numItermax=numInnerItermax, verbose=verbose, log=log, polytope_vertices=polytope_vertices, **kwargs)
        # check Nan
        if torch.isnan(T_sum).any():
            print('Nan detected, stop iteration')
            break

        if annealing:
            step_radius *= (1 - eps_annealing)
            probe_radius *= (1 - eps_annealing)
        # update log
        # if log:
        #     update_inner_log_dict(log_dict, inner_log_dict)

        displacement_square_norm = torch.square(T_sum - X).sum()
        if update_start and update_end:
            X = T_sum
        else:
            traj_len = traj_dim[-2]
            X = X.view(traj_dim)
            T_sum = T_sum.view(traj_dim)
            X[..., 1:traj_len - 1, :] = T_sum[..., 1:traj_len - 1, :]
            if update_start:
                X[..., 0, :] = T_sum[..., 0, :]
            if update_end:
                X[..., traj_len - 1, :] = T_sum[..., traj_len - 1, :]
            X = X.view(k, d)
        if verbose:
            print(f'Iteration {iter_count}: displacement_square_norm={displacement_square_norm}')
        iter_count += 1
        if log:
            X_hist.append(X.clone())
            displacement_square_norms.append(displacement_square_norm)

    if log:
        log_dict['outer_iter'] = iter_count
        log_dict['displacement_square_norms'] = displacement_square_norms
        log_dict['X_hist'] = X_hist
    return X, log_dict

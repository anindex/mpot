import torch
import warnings
from functools import partial


def compute_K(M, a, b, reg):
        """log space computation"""
        return torch.exp(-(M - a.unsqueeze(1) - b.unsqueeze(0)) / reg)


def compute_pi(M, a, b, u, v, reg):
    """log space gamma computation"""
    return torch.exp(-(M - a.unsqueeze(1) - b.unsqueeze(0)) / reg + torch.log(u.unsqueeze(1)) + torch.log(v.unsqueeze(0)))


def sinkhorn_knopp_stabilized(alpha, beta, M, reg, numItermax=1000, tau=1e3, stopThr=1e-5,
                              verbose=False, print_period=1, log=False, warn=False, **kwargs):
    reg = torch.tensor(reg).type_as(M)
    if len(alpha) == 0:
        alpha = torch.full((M.shape[0],), 1 / M.shape[0]).type_as(M)
    if len(beta) == 0:
        beta = torch.full((M.shape[1],), 1 / M.shape[1]).type_as(M)

    # init data
    dim_a, dim_b = M.shape
    log_dict = {'err': []}

    a, b = torch.zeros(dim_a).type_as(M), torch.zeros(dim_b).type_as(M)
    u, v = torch.full((dim_a, ), 1 / dim_a).type_as(M) , torch.full((dim_b, ), 1 / dim_b).type_as(M)
    K = compute_K(M, a, b, reg)
    transp = K
    err = 1

    for ii in range(numItermax):
        uprev = u
        vprev = v

        # sinkhorn-knopp iteration
        v = beta / (K.T @ u)
        u = alpha / (K @ v)

        # remove numerical problems and store them in K
        if torch.max(torch.abs(u)) > tau or torch.max(torch.abs(v)) > tau:
            a, b = a + reg * torch.log(u), b + reg * torch.log(v)
            u, v = torch.full((dim_a, ), 1 / dim_a).type_as(M) , torch.full((dim_b, ), 1 / dim_b).type_as(M)
            K = compute_K(M, a, b, reg)

        if ii % print_period == 0:
            transp = compute_pi(M, a, b, u, v, reg)
            err = torch.norm(torch.sum(transp, axis=0) - beta)
            if log:
                log_dict['err'].append(err.cpu().numpy())

            if verbose:
                if ii % (print_period * 20) == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))
            if err <= stopThr:
                break

        if torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)):
            # we have reached the machine precision, break the loop and return last solution
            warnings.warn('Numerical errors at iteration %d' % ii)
            u = uprev
            v = vprev
            break
    else:
        if warn:
            warnings.warn("Sinkhorn did not converge. Try to increase the numItermax or reg.")
    if log:
        logu = a / reg + torch.log(u)
        logv = b / reg + torch.log(v)
        log_dict['inner_iter'] = ii
        log_dict['logu'] = logu.cpu().numpy()
        log_dict['logv'] = logv.cpu().numpy()
        log_dict['a'] = (a + reg * torch.log(u)).cpu().numpy()
        log_dict['b'] = (b + reg * torch.log(v)).cpu().numpy()
        return compute_pi(M, a, b, u, v, reg), log_dict
    else:
        return compute_pi(M, a, b, u, v, reg), log_dict

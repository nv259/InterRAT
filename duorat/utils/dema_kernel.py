import numpy as np
import torch


def get_kernel(params: torch.Tensor, num_particles):
    """
    Compute the RBF kernel for the input

    Args:
        params: a tensor of shape (N, M)

    Returns: kernel_matrix = tensor of shape (N, N)
    """
    pairwise_d_matrix = get_pairwise_distance_matrix(x=params)

    median_dist = torch.quantile(
        input=pairwise_d_matrix, q=0.5
    )  # tf.reduce_mean(euclidean_dists) ** 2
    h = median_dist / np.log(num_particles)

    kernel_matrix = torch.exp(-pairwise_d_matrix / h)
    kernel_sum = torch.sum(input=kernel_matrix, dim=1, keepdim=True)
    grad_kernel = -torch.matmul(kernel_matrix, params)
    grad_kernel += params * kernel_sum
    grad_kernel /= h

    return kernel_matrix, grad_kernel, h


def get_kernel_wSGLD_B(params: torch.Tensor, num_particles):
    """
    Compute the RBF kernel and repulsive term for wSGLD

    Args:
        params: a tensor of shape (N, M)

    Returns:
        - kernel_matrix = tensor of shape (N, N)
        - repulsive term
    """
    pairwise_d_matrix = get_pairwise_distance_matrix(x=params)
    median_dist = torch.quantile(input=pairwise_d_matrix, q=0.5)
    h = median_dist / np.log(num_particles)
    kernel_matrix = torch.exp(-pairwise_d_matrix / h)
    kernel_sum = torch.sum(input=kernel_matrix, dim=1, keepdim=True)
    # compute repulsive term of w_SGLD_B
    # invert of kernel_sum Nx1
    invert_kernel_sum = kernel_sum.pow_(-1)
    grad_kernel = params * (
        torch.matmul(kernel_matrix, invert_kernel_sum)
        + torch.sum(kernel_matrix * invert_kernel_sum, dim=1, keepdim=True)
    )
    grad_kernel += -(
        torch.matmul(
            kernel_matrix * torch.transpose(invert_kernel_sum, 0, 1), params
        )
        + torch.matmul(kernel_matrix, params) * invert_kernel_sum
    )
    grad_kernel /= h

    return kernel_matrix, grad_kernel, h


def get_pairwise_distance_matrix(x: torch.Tensor) -> torch.Tensor:
    """Calculate the pairwise distance between each row of tensor x

    Args:
        x: input tensor

    Return: matrix of point-wise distances
    """
    n, m = x.shape

    # initialize matrix of pairwise distances as a N x N matrix
    pairwise_d_matrix = torch.zeros(size=(n, n), device=x.device)

    # num_particles = particle_tensor.shape[0]
    euclidean_dists = torch.nn.functional.pdist(input=x, p=2)  # shape of (N)

    # assign upper-triangle part
    triu_indices = torch.triu_indices(row=n, col=n, offset=1)
    pairwise_d_matrix[triu_indices[0], triu_indices[1]] = euclidean_dists

    # assign lower-triangle part
    pairwise_d_matrix = torch.transpose(pairwise_d_matrix, dim0=0, dim1=1)
    pairwise_d_matrix[triu_indices[0], triu_indices[1]] = euclidean_dists

    return pairwise_d_matrix
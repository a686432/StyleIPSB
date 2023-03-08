""" Use scipy/ARPACK implicitly restarted lanczos to find top k eigenthings
This code solve generalized eigenvalue problem for operators or matrices
Format adapted from lanczos.py in hessian-eigenthings
"""
import torch
import numpy as np
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator
from scipy.sparse.linalg import eigsh
from warnings import warn

def lanczos_generalized(
    operator,
    metric_operator=None,
    metric_inv_operator=None,
    num_eigenthings=10,
    which="LM",
    max_steps=20,
    tol=1e-6,
    num_lanczos_vectors=None,
    init_vec=None,
    use_gpu=False,
):
    """
    Use the scipy.sparse.linalg.eigsh hook to the ARPACK lanczos algorithm
    to find the top k eigenvalues/eigenvectors for the *generalized eigen problem*. 
        A v = lambda B v
    B is the metric tensor of the linear space, while A is the linear function on it. 

    Parameters
    -------------
    operator: power_iter.Operator
        linear operator A to solve.
    metric_operator: Operator, linear operator B. 
    metric_inv_operator: Operator, linear operator B^{-1}. 
    num_eigenthings : int
        number of eigenvalue/eigenvector pairs to compute
    which : str ['LM', SM', 'LA', SA']
        L,S = largest, smallest. M, A = in magnitude, algebriac
        SM = smallest in magnitude. LA = largest algebraic.
    max_steps : int
        maximum number of arnoldi updates
    tol : float
        relative accuracy of eigenvalues / stopping criterion
    num_lanczos_vectors : int
        number of lanczos vectors to compute. if None, > 2*num_eigenthings
    init_vec: [torch.Tensor, torch.cuda.Tensor]
        if None, use random tensor. this is the init vec for arnoldi updates.
    use_gpu: bool
        if true, use cuda tensors.

    Returns
    ----------------
    eigenvalues : np.ndarray
        array containing `num_eigenthings` eigenvalues of the operator
    eigenvectors : np.ndarray
        array containing `num_eigenthings` eigenvectors of the operator
    """
    if isinstance(operator.size, int):
        size = operator.size
    else:
        size = operator.size[0]
    shape = (size, size)

    if num_lanczos_vectors is None:
        num_lanczos_vectors = min(2 * num_eigenthings, size - 1)
    if num_lanczos_vectors < 2 * num_eigenthings:
        warn(
            "[lanczos] number of lanczos vectors should usually be > 2*num_eigenthings"
        )

    def _scipy_apply(x):
        x = torch.from_numpy(x)
        if use_gpu:
            x = x.cuda()
        return operator.apply(x.float()).cpu().numpy()
    scipy_op = ScipyLinearOperator(shape, _scipy_apply)

    if isinstance(metric_operator, np.ndarray) or \
       isinstance(metric_operator, ScipyLinearOperator):
        metric_op = metric_operator
    else:
        def _scipy_apply_metric(x):
            x = torch.from_numpy(x)
            if use_gpu:
                x = x.cuda()
            return metric_operator.apply(x.float()).cpu().numpy()
        metric_op = ScipyLinearOperator(shape, _scipy_apply_metric)

    if isinstance(metric_inv_operator, np.ndarray) or \
       isinstance(metric_inv_operator, ScipyLinearOperator):
        metric_inv_op = metric_inv_operator
    else:
        def _scipy_apply_metric_inv(x):
            x = torch.from_numpy(x)
            if use_gpu:
                x = x.cuda()
            return metric_inv_operator.apply(x.float()).cpu().numpy()
        metric_inv_op = ScipyLinearOperator(shape, _scipy_apply_metric_inv)

    if init_vec is None:
        init_vec = np.random.rand(size)
    elif isinstance(init_vec, torch.Tensor):
        init_vec = init_vec.cpu().numpy()
    eigenvals, eigenvecs = eigsh(
        A=scipy_op,
        k=num_eigenthings,
        M=metric_op,
        Minv=metric_inv_op,
        which=which,
        maxiter=max_steps,
        tol=tol,
        ncv=num_lanczos_vectors,
        return_eigenvectors=True,
    )
    return eigenvals, eigenvecs.T


def lanczos(
    operator,
    num_eigenthings=10,
    which="LM",
    max_steps=20,
    tol=1e-6,
    num_lanczos_vectors=None,
    init_vec=None,
    use_gpu=False,
):
    """
    https://github.com/noahgolmant/pytorch-hessian-eigenthings/blob/8ff8b3907f2383fe1fdaa232736c8fef295d8131/hessian_eigenthings/lanczos.py#L11
    Use the scipy.sparse.linalg.eigsh hook to the ARPACK lanczos algorithm
    to find the top k eigenvalues/eigenvectors.
    Parameters
    -------------
    operator: power_iter.Operator
        linear operator to solve.
    num_eigenthings : int
        number of eigenvalue/eigenvector pairs to compute
    which : str ['LM', SM', 'LA', SA']
        L,S = largest, smallest. M, A = in magnitude, algebriac
        SM = smallest in magnitude. LA = largest algebraic.
    max_steps : int
        maximum number of arnoldi updates
    tol : float
        relative accuracy of eigenvalues / stopping criterion
    num_lanczos_vectors : int
        number of lanczos vectors to compute. if None, > 2*num_eigenthings
    init_vec: [torch.Tensor, torch.cuda.Tensor]
        if None, use random tensor. this is the init vec for arnoldi updates.
    use_gpu: bool
        if true, use cuda tensors.
    Returns
    ----------------
    eigenvalues : np.ndarray
        array containing `num_eigenthings` eigenvalues of the operator
    eigenvectors : np.ndarray
        array containing `num_eigenthings` eigenvectors of the operator
    """
    if isinstance(operator.size, int):
        size = operator.size
    else:
        size = operator.size[0]
    shape = (size, size)

    if num_lanczos_vectors is None:
        num_lanczos_vectors = min(2 * num_eigenthings, size - 1)
    if num_lanczos_vectors < 2 * num_eigenthings:
        warn(
            "[lanczos] number of lanczos vectors should usually be > 2*num_eigenthings"
        )

    def _scipy_apply(x):
        x = torch.from_numpy(x)
        if use_gpu:
            x = x.cuda()
        out = operator.apply(x.float())
        out = out.cpu().numpy()
        return out

    scipy_op = ScipyLinearOperator(shape, _scipy_apply)
    if init_vec is None:
        init_vec = np.random.rand(size)
    elif isinstance(init_vec, torch.Tensor):
        init_vec = init_vec.cpu().numpy()

    eigenvals, eigenvecs = eigsh(
        A=scipy_op,
        k=num_eigenthings,
        which=which,
        maxiter=max_steps,
        tol=tol,
        ncv=num_lanczos_vectors,
        return_eigenvectors=True,
    )
    return eigenvals, eigenvecs.T

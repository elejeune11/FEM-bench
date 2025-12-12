def MSA_3D_solve_eigenvalue_CC1_H1_T1(K_e_global: np.ndarray, K_g_global: np.ndarray, boundary_conditions: dict, n_nodes: int):
    """
    Compute the smallest positive elastic critical load factor and corresponding
    global buckling mode shape for a 3D frame/beam model.
    The generalized eigenproblem is solved on the free DOFs:
        K_e_ff * phi = -lambda * K_g_ff * phi
    where K_e_ff and K_g_ff are the partitions of the elastic and geometric
    stiffness matrices after applying boundary conditions.
    Parameters
    ----------
    K_e_global : ndarray, shape (6*n_nodes, 6*n_nodes)
        Global elastic stiffness matrix.
    K_g_global : ndarray, shape (6*n_nodes, 6*n_nodes)
        Global geometric stiffness matrix at the reference load state.
    boundary_conditions : dict[int, array-like of bool]
        Dictionary mapping each node index (0-based) to a 6-element boolean array
        defining the constrained DOFs:
            True  → fixed (prescribed zero displacement/rotation)
            False → free (unknown)
        Nodes not listed are assumed fully free.
    n_nodes : int
        Number of nodes in the model (assumes 6 DOFs per node, ordered
        [u_x, u_y, u_z, theta_x, theta_y, theta_z] per node).
    Returns
    -------
    elastic_critical_load_factor : float
        The smallest positive eigenvalue lambda (> 0), interpreted as the elastic
        critical load factor (i.e., P_cr = lambda * P_ref, if K_g_global was
        formed at reference load P_ref).
    deformed_shape_vector : ndarray, shape (6*n_nodes,)
        Global buckling mode vector with entries on constrained DOFs set to zero.
        No normalization is applied (matches original behavior).
    Helper Functions
    ----------------
        Identifies which global DOFs are fixed and which are free, returning
        sorted integer index arrays (`fixed`, `free`). This helper ensures
        consistency between the nodal boundary-condition specification and the
        DOF layout assumed here.
    Raises
    ------
    ValueError
            Use a tolerence of 1e16
    """
    import numpy as np
    import scipy
    import pytest
    from typing import Callable
    (fixed, free) = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    n_dof = 6 * n_nodes
    if free.size == 0:
        raise ValueError('No positive eigenvalue is found.')
    K_e_ff = K_e_global[np.ix_(free, free)]
    K_g_ff = K_g_global[np.ix_(free, free)]
    tol_cond = 1e+16

    def _cond_num(M):
        try:
            return np.linalg.cond(M)
        except Exception:
            return np.inf
    cond_e = _cond_num(K_e_ff)
    cond_g = _cond_num(K_g_ff)
    if not np.isfinite(cond_e) or cond_e > tol_cond or (not np.isfinite(cond_g)) or (cond_g > tol_cond):
        raise ValueError('The reduced matrices are ill-conditioned/singular beyond tolerance.')
    try:
        (w, v) = scipy.linalg.eig(K_e_ff, -K_g_ff, check_finite=True)
    except Exception as exc:
        raise ValueError('Generalized eigenvalue solve failed.') from exc
    w_real = np.real(w)
    w_imag = np.imag(w)
    finite_mask = np.isfinite(w_real) & np.isfinite(w_imag)
    complex_tol = 1e-08
    real_mask = np.abs(w_imag) <= complex_tol * np.maximum(1.0, np.abs(w_real))
    positive_mask = w_real > 0.0
    candidates_mask = finite_mask & real_mask & positive_mask
    if not np.any(candidates_mask):
        if np.any(finite_mask & positive_mask & ~real_mask):
            raise ValueError('Eigenpairs contain non-negligible complex parts.')
        raise ValueError('No positive eigenvalue is found.')
    idxs = np.where(candidates_mask)[0]
    idx_min = idxs[np.argmin(w_real[idxs])]
    lambda_min = w_real[idx_min]
    vec = v[:, idx_min]
    v_imag_max = float(np.max(np.abs(np.imag(vec)))) if vec.size else 0.0
    v_real_max = float(np.max(np.abs(np.real(vec)))) if vec.size else 0.0
    if v_imag_max > complex_tol * max(1.0, v_real_max):
        k = int(np.argmax(np.abs(vec))) if vec.size else 0
        phase = np.angle(vec[k]) if vec.size else 0.0
        vec_rot = vec * np.exp(-1j * phase)
        v_imag_max2 = float(np.max(np.abs(np.imag(vec_rot)))) if vec_rot.size else 0.0
        v_real_max2 = float(np.max(np.abs(np.real(vec_rot)))) if vec_rot.size else 0.0
        if v_imag_max2 > complex_tol * max(1.0, v_real_max2):
            raise ValueError('Eigenpairs contain non-negligible complex parts.')
        mode_free = np.real(vec_rot)
    else:
        mode_free = np.real(vec)
    deformed = np.zeros(n_dof, dtype=float)
    deformed[free] = mode_free
    return (float(lambda_min), deformed)
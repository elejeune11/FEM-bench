def tri6_shape_functions_and_derivatives(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized evaluation of quadratic (6-node) triangular shape functions and derivatives.
    Parameters
    ----------
    xi : np.ndarray
        Natural coordinates in the reference triangle.
    Returns
    -------
    N : np.ndarray
        Shape functions evaluated at the input points. Shape: (n, 6, 1).
        Node order: [N1, N2, N3, N4, N5, N6].
    dN_dxi : np.ndarray
        Partial derivatives w.r.t (ξ, η). Shape: (n, 6, 2).
        Columns correspond to [∂()/∂ξ, ∂()/∂η] in the same node order.
    Raises
    ------
    ValueError
        If `xi` is not a NumPy array.
        If `xi` has shape other than (2,) or (n, 2).
        If `xi` contains non-finite values (NaN or Inf).
    Notes
    -----
    Uses P2 triangle with ξ_c = 1 - ξ - η:
        N1 = ξ(2ξ - 1),   N2 = η(2η - 1),   N3 = ξ_c(2ξ_c - 1),
        N4 = 4ξη,         N5 = 4ηξ_c,       N6 = 4ξξ_c.
    """
    if not isinstance(xi, np.ndarray):
        raise ValueError('`xi` must be a NumPy array.')
    if xi.ndim not in [1, 2] or (xi.ndim == 1 and xi.shape != (2,)) or (xi.ndim == 2 and xi.shape[1] != 2):
        raise ValueError('`xi` must have shape (2,) or (n, 2).')
    if not np.all(np.isfinite(xi)):
        raise ValueError('`xi` must contain finite values.')
    xi_proc = np.atleast_2d(xi)
    n_points = xi_proc.shape[0]
    xi_ = xi_proc[:, 0]
    eta = xi_proc[:, 1]
    zeta = 1.0 - xi_ - eta
    N = np.zeros((n_points, 6))
    N[:, 0] = xi_ * (2.0 * xi_ - 1.0)
    N[:, 1] = eta * (2.0 * eta - 1.0)
    N[:, 2] = zeta * (2.0 * zeta - 1.0)
    N[:, 3] = 4.0 * xi_ * eta
    N[:, 4] = 4.0 * eta * zeta
    N[:, 5] = 4.0 * xi_ * zeta
    N_out = np.expand_dims(N, axis=2)
    dN_dxi = np.zeros((n_points, 6, 2))
    dN_dxi[:, 0, 0] = 4.0 * xi_ - 1.0
    dN_dxi[:, 2, 0] = 1.0 - 4.0 * zeta
    dN_dxi[:, 3, 0] = 4.0 * eta
    dN_dxi[:, 4, 0] = -4.0 * eta
    dN_dxi[:, 5, 0] = 4.0 * (zeta - xi_)
    dN_dxi[:, 1, 1] = 4.0 * eta - 1.0
    dN_dxi[:, 2, 1] = 1.0 - 4.0 * zeta
    dN_dxi[:, 3, 1] = 4.0 * xi_
    dN_dxi[:, 4, 1] = 4.0 * (zeta - eta)
    dN_dxi[:, 5, 1] = -4.0 * xi_
    return (N_out, dN_dxi)
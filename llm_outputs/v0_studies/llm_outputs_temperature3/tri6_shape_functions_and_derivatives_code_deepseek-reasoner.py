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
        raise ValueError('xi must be a NumPy array')
    if xi.ndim == 1:
        if xi.shape[0] != 2:
            raise ValueError('For 1D input, xi must have shape (2,)')
        xi = xi.reshape(1, 2)
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('For 2D input, xi must have shape (n, 2)')
    else:
        raise ValueError('xi must have shape (2,) or (n, 2)')
    if not np.all(np.isfinite(xi)):
        raise ValueError('xi contains non-finite values')
    n_points = xi.shape[0]
    xi_vals = xi[:, 0]
    eta_vals = xi[:, 1]
    xi_c_vals = 1.0 - xi_vals - eta_vals
    N = np.zeros((n_points, 6))
    N[:, 0] = xi_vals * (2.0 * xi_vals - 1.0)
    N[:, 1] = eta_vals * (2.0 * eta_vals - 1.0)
    N[:, 2] = xi_c_vals * (2.0 * xi_c_vals - 1.0)
    N[:, 3] = 4.0 * xi_vals * eta_vals
    N[:, 4] = 4.0 * eta_vals * xi_c_vals
    N[:, 5] = 4.0 * xi_vals * xi_c_vals
    dN_dxi = np.zeros((n_points, 6, 2))
    dN_dxi[:, 0, 0] = 4.0 * xi_vals - 1.0
    dN_dxi[:, 0, 1] = 0.0
    dN_dxi[:, 1, 0] = 0.0
    dN_dxi[:, 1, 1] = 4.0 * eta_vals - 1.0
    dN_dxi[:, 2, 0] = 1.0 - 4.0 * xi_c_vals
    dN_dxi[:, 2, 1] = 1.0 - 4.0 * xi_c_vals
    dN_dxi[:, 3, 0] = 4.0 * eta_vals
    dN_dxi[:, 3, 1] = 4.0 * xi_vals
    dN_dxi[:, 4, 0] = -4.0 * eta_vals
    dN_dxi[:, 4, 1] = 4.0 * (xi_c_vals - eta_vals)
    dN_dxi[:, 5, 0] = 4.0 * (xi_c_vals - xi_vals)
    dN_dxi[:, 5, 1] = -4.0 * xi_vals
    return (N.reshape(n_points, 6, 1), dN_dxi)
def FEM_2D_tri6_shape_fcns_and_derivatives_CC0_H0_T0(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        raise ValueError('xi must be a numpy array')
    if xi.ndim == 1:
        if xi.shape != (2,):
            raise ValueError('xi must have shape (2,) or (n, 2)')
        arr = xi.reshape(1, 2).astype(float)
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('xi must have shape (2,) or (n, 2)')
        arr = xi.astype(float)
    else:
        raise ValueError('xi must have shape (2,) or (n, 2)')
    if not np.isfinite(arr).all():
        raise ValueError('xi contains non-finite values')
    xi_vals = arr[:, 0]
    eta_vals = arr[:, 1]
    xi_c = 1.0 - xi_vals - eta_vals
    N1 = xi_vals * (2.0 * xi_vals - 1.0)
    N2 = eta_vals * (2.0 * eta_vals - 1.0)
    N3 = xi_c * (2.0 * xi_c - 1.0)
    N4 = 4.0 * xi_vals * eta_vals
    N5 = 4.0 * eta_vals * xi_c
    N6 = 4.0 * xi_vals * xi_c
    N = np.stack((N1, N2, N3, N4, N5, N6), axis=1)
    N = N.reshape(N.shape[0], 6, 1)
    dN_dxi = np.empty((arr.shape[0], 6, 2), dtype=float)
    dN_dxi[:, 0, 0] = 4.0 * xi_vals - 1.0
    dN_dxi[:, 0, 1] = 0.0
    dN_dxi[:, 1, 0] = 0.0
    dN_dxi[:, 1, 1] = 4.0 * eta_vals - 1.0
    dN_dxi[:, 2, 0] = -(4.0 * xi_c - 1.0)
    dN_dxi[:, 2, 1] = -(4.0 * xi_c - 1.0)
    dN_dxi[:, 3, 0] = 4.0 * eta_vals
    dN_dxi[:, 3, 1] = 4.0 * xi_vals
    dN_dxi[:, 4, 0] = -4.0 * eta_vals
    dN_dxi[:, 4, 1] = 4.0 * (1.0 - xi_vals - 2.0 * eta_vals)
    dN_dxi[:, 5, 0] = 4.0 * (1.0 - 2.0 * xi_vals - eta_vals)
    dN_dxi[:, 5, 1] = -4.0 * xi_vals
    return (N, dN_dxi)
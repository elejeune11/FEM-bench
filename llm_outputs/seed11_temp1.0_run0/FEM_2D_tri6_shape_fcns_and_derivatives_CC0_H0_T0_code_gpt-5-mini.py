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
        raise ValueError('xi must be a NumPy array.')
    if xi.ndim == 1:
        if xi.shape != (2,):
            raise ValueError('xi must have shape (2,) or (n, 2).')
        xi_arr = xi.reshape(1, 2).astype(float)
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('xi must have shape (2,) or (n, 2).')
        xi_arr = xi.astype(float)
    else:
        raise ValueError('xi must have shape (2,) or (n, 2).')
    if not np.isfinite(xi_arr).all():
        raise ValueError('xi contains non-finite values.')
    xi1 = xi_arr[:, 0]
    eta = xi_arr[:, 1]
    s = 1.0 - xi1 - eta
    N1 = xi1 * (2.0 * xi1 - 1.0)
    N2 = eta * (2.0 * eta - 1.0)
    N3 = s * (2.0 * s - 1.0)
    N4 = 4.0 * xi1 * eta
    N5 = 4.0 * eta * s
    N6 = 4.0 * xi1 * s
    n = xi_arr.shape[0]
    N = np.zeros((n, 6, 1), dtype=float)
    N[:, 0, 0] = N1
    N[:, 1, 0] = N2
    N[:, 2, 0] = N3
    N[:, 3, 0] = N4
    N[:, 4, 0] = N5
    N[:, 5, 0] = N6
    dN = np.zeros((n, 6, 2), dtype=float)
    dN[:, 0, 0] = 4.0 * xi1 - 1.0
    dN[:, 0, 1] = 0.0
    dN[:, 1, 0] = 0.0
    dN[:, 1, 1] = 4.0 * eta - 1.0
    common3 = 4.0 * s - 1.0
    dN[:, 2, 0] = -common3
    dN[:, 2, 1] = -common3
    dN[:, 3, 0] = 4.0 * eta
    dN[:, 3, 1] = 4.0 * xi1
    dN[:, 4, 0] = -4.0 * eta
    dN[:, 4, 1] = 4.0 * (s - eta)
    dN[:, 5, 0] = 4.0 * (s - xi1)
    dN[:, 5, 1] = -4.0 * xi1
    return (N, dN)
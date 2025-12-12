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
        raise ValueError('`xi` must be a NumPy array.')
    if xi.ndim == 1:
        if xi.shape != (2,):
            raise ValueError('`xi` with 1 dimension must have shape (2,).')
        xi = xi.reshape(1, 2)
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('`xi` with 2 dimensions must have shape (n, 2).')
    else:
        raise ValueError('`xi` must have shape (2,) or (n, 2).')
    if not np.all(np.isfinite(xi)):
        raise ValueError('`xi` must contain finite values.')
    n = xi.shape[0]
    xsi = xi[:, 0]
    eta = xi[:, 1]
    xsi_c = 1.0 - xsi - eta
    N = np.zeros((n, 6, 1))
    N[:, 0, 0] = xsi * (2 * xsi - 1)
    N[:, 1, 0] = eta * (2 * eta - 1)
    N[:, 2, 0] = xsi_c * (2 * xsi_c - 1)
    N[:, 3, 0] = 4 * xsi * eta
    N[:, 4, 0] = 4 * eta * xsi_c
    N[:, 5, 0] = 4 * xsi * xsi_c
    dN_dxi = np.zeros((n, 6, 2))
    dN_dxi[:, 0, 0] = 4 * xsi - 1
    dN_dxi[:, 2, 0] = 1 - 4 * xsi_c
    dN_dxi[:, 3, 0] = 4 * eta
    dN_dxi[:, 4, 0] = -4 * eta
    dN_dxi[:, 5, 0] = 4 * (xsi_c - xsi)
    dN_dxi[:, 1, 1] = 4 * eta - 1
    dN_dxi[:, 2, 1] = 1 - 4 * xsi_c
    dN_dxi[:, 3, 1] = 4 * xsi
    dN_dxi[:, 4, 1] = 4 * (xsi_c - eta)
    dN_dxi[:, 5, 1] = -4 * xsi
    return (N, dN_dxi)
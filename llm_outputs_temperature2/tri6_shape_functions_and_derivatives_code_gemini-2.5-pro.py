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
        raise ValueError('Input `xi` must be a NumPy array.')
    if xi.ndim == 1:
        if xi.shape == (2,):
            xi = xi.reshape(1, 2)
        else:
            raise ValueError('Input `xi` with 1 dimension must have shape (2,).')
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('Input `xi` with 2 dimensions must have shape (n, 2).')
    else:
        raise ValueError('Input `xi` must have 1 or 2 dimensions.')
    if not np.all(np.isfinite(xi)):
        raise ValueError('Input `xi` must contain finite values.')
    n = xi.shape[0]
    ksi = xi[:, 0]
    eta = xi[:, 1]
    ksi_c = 1.0 - ksi - eta
    N1 = ksi * (2 * ksi - 1)
    N2 = eta * (2 * eta - 1)
    N3 = ksi_c * (2 * ksi_c - 1)
    N4 = 4 * ksi * eta
    N5 = 4 * eta * ksi_c
    N6 = 4 * ksi * ksi_c
    N = np.stack([N1, N2, N3, N4, N5, N6], axis=1)
    N = N[:, :, np.newaxis]
    dN_dksi = np.zeros((n, 6))
    dN_dksi[:, 0] = 4 * ksi - 1
    dN_dksi[:, 2] = 1 - 4 * ksi_c
    dN_dksi[:, 3] = 4 * eta
    dN_dksi[:, 4] = -4 * eta
    dN_dksi[:, 5] = 4 * ksi_c - 4 * ksi
    dN_deta = np.zeros((n, 6))
    dN_deta[:, 1] = 4 * eta - 1
    dN_deta[:, 2] = 1 - 4 * ksi_c
    dN_deta[:, 3] = 4 * ksi
    dN_deta[:, 4] = 4 * ksi_c - 4 * eta
    dN_deta[:, 5] = -4 * ksi
    dN_dxi = np.stack([dN_dksi, dN_deta], axis=2)
    return (N, dN_dxi)
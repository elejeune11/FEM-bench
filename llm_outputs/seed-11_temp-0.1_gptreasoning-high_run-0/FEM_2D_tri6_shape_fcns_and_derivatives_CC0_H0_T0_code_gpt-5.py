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
        xi2d = xi[np.newaxis, :]
    elif xi.ndim == 2 and xi.shape[1] == 2:
        xi2d = xi
    else:
        raise ValueError('xi must have shape (2,) or (n, 2).')
    if not np.all(np.isfinite(xi2d)):
        raise ValueError('xi contains non-finite values (NaN or Inf).')
    x = xi2d[:, 0]
    y = xi2d[:, 1]
    s = 1.0 - x - y
    N1 = x * (2.0 * x - 1.0)
    N2 = y * (2.0 * y - 1.0)
    N3 = s * (2.0 * s - 1.0)
    N4 = 4.0 * x * y
    N5 = 4.0 * y * s
    N6 = 4.0 * x * s
    N = np.stack([N1, N2, N3, N4, N5, N6], axis=1)[..., np.newaxis]
    zeros = np.zeros_like(x)
    dNdx = np.stack([4.0 * x - 1.0, zeros, 1.0 - 4.0 * s, 4.0 * y, -4.0 * y, 4.0 * (s - x)], axis=1)
    dNdy = np.stack([zeros, 4.0 * y - 1.0, 1.0 - 4.0 * s, 4.0 * x, 4.0 * (s - y), -4.0 * x], axis=1)
    dN_dxi = np.stack([dNdx, dNdy], axis=2)
    return (N, dN_dxi)
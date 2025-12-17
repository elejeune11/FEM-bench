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
            raise ValueError('`xi` must have shape (2,) for a single point or (n, 2) for multiple points.')
        xi2d = xi.reshape(1, 2)
    elif xi.ndim == 2 and xi.shape[1] == 2:
        xi2d = xi
    else:
        raise ValueError('`xi` must have shape (2,) for a single point or (n, 2) for multiple points.')
    if not np.all(np.isfinite(xi2d)):
        raise ValueError('`xi` contains non-finite values (NaN or Inf).')
    s = xi2d[:, 0].astype(np.float64, copy=False)
    t = xi2d[:, 1].astype(np.float64, copy=False)
    r = 1.0 - s - t
    N1 = s * (2.0 * s - 1.0)
    N2 = t * (2.0 * t - 1.0)
    N3 = r * (2.0 * r - 1.0)
    N4 = 4.0 * s * t
    N5 = 4.0 * t * r
    N6 = 4.0 * s * r
    N = np.stack([N1, N2, N3, N4, N5, N6], axis=1)[:, :, None]
    dN1_dxi = 4.0 * s - 1.0
    dN2_dxi = np.zeros_like(s)
    dN3_dxi = -4.0 * r + 1.0
    dN4_dxi = 4.0 * t
    dN5_dxi = -4.0 * t
    dN6_dxi = 4.0 * r - 4.0 * s
    dN1_deta = np.zeros_like(t)
    dN2_deta = 4.0 * t - 1.0
    dN3_deta = -4.0 * r + 1.0
    dN4_deta = 4.0 * s
    dN5_deta = 4.0 * r - 4.0 * t
    dN6_deta = -4.0 * s
    dXi = np.stack([dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi, dN5_dxi, dN6_dxi], axis=1)
    dEta = np.stack([dN1_deta, dN2_deta, dN3_deta, dN4_deta, dN5_deta, dN6_deta], axis=1)
    dN_dxi = np.stack([dXi, dEta], axis=2)
    return (N, dN_dxi)
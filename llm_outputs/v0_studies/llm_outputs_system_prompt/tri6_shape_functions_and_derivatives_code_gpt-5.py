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
    if xi.ndim == 1 and xi.shape == (2,):
        xi_arr = xi.reshape(1, 2)
    elif xi.ndim == 2 and xi.shape[1] == 2:
        xi_arr = xi
    else:
        raise ValueError('`xi` must have shape (2,) or (n, 2).')
    if not np.all(np.isfinite(xi_arr)):
        raise ValueError('`xi` contains non-finite values.')
    r = xi_arr[:, 0]
    s = xi_arr[:, 1]
    t = 1.0 - r - s
    N1 = r * (2.0 * r - 1.0)
    N2 = s * (2.0 * s - 1.0)
    N3 = t * (2.0 * t - 1.0)
    N4 = 4.0 * r * s
    N5 = 4.0 * s * t
    N6 = 4.0 * r * t
    N = np.stack([N1, N2, N3, N4, N5, N6], axis=1)[..., np.newaxis]
    dN1_dr = 4.0 * r - 1.0
    dN2_dr = np.zeros_like(r, dtype=float)
    dN3_dr = -4.0 * t + 1.0
    dN4_dr = 4.0 * s
    dN5_dr = -4.0 * s
    dN6_dr = 4.0 * t - 4.0 * r
    dN1_ds = np.zeros_like(s, dtype=float)
    dN2_ds = 4.0 * s - 1.0
    dN3_ds = -4.0 * t + 1.0
    dN4_ds = 4.0 * r
    dN5_ds = 4.0 * t - 4.0 * s
    dN6_ds = -4.0 * r
    dNdr = np.stack([dN1_dr, dN2_dr, dN3_dr, dN4_dr, dN5_dr, dN6_dr], axis=1)
    dNds = np.stack([dN1_ds, dN2_ds, dN3_ds, dN4_ds, dN5_ds, dN6_ds], axis=1)
    dN_dxi = np.stack([dNdr, dNds], axis=2)
    return (N, dN_dxi)
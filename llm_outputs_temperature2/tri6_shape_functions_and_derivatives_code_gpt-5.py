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
        raise ValueError('xi must be a NumPy array.')
    if xi.ndim == 1:
        if xi.shape != (2,):
            raise ValueError('xi must have shape (2,) or (n, 2).')
        Xi = xi.reshape(1, 2)
    elif xi.ndim == 2 and xi.shape[1] == 2:
        Xi = xi
    else:
        raise ValueError('xi must have shape (2,) or (n, 2).')
    if not np.isfinite(Xi).all():
        raise ValueError('xi must contain only finite values.')
    Xi = Xi.astype(np.float64, copy=False)
    s = Xi[:, 0]
    t = Xi[:, 1]
    xc = 1.0 - s - t
    N1 = s * (2.0 * s - 1.0)
    N2 = t * (2.0 * t - 1.0)
    N3 = xc * (2.0 * xc - 1.0)
    N4 = 4.0 * s * t
    N5 = 4.0 * t * xc
    N6 = 4.0 * s * xc
    N = np.stack([N1, N2, N3, N4, N5, N6], axis=1)[..., np.newaxis]
    zero = np.zeros_like(s)
    dN1 = np.stack([4.0 * s - 1.0, zero], axis=1)
    dN2 = np.stack([zero, 4.0 * t - 1.0], axis=1)
    dN3 = np.stack([-4.0 * xc + 1.0, -4.0 * xc + 1.0], axis=1)
    dN4 = np.stack([4.0 * t, 4.0 * s], axis=1)
    dN5 = np.stack([-4.0 * t, 4.0 * (xc - t)], axis=1)
    dN6 = np.stack([4.0 * (xc - s), -4.0 * s], axis=1)
    dN_dxi = np.stack([dN1, dN2, dN3, dN4, dN5, dN6], axis=1)
    return (N, dN_dxi)
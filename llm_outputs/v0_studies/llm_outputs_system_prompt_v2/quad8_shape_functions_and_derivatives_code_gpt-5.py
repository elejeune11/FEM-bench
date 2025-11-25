def quad8_shape_functions_and_derivatives(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized evaluation of quadratic (8-node) quadrilateral shape functions and derivatives.
    Parameters
    ----------
    xi : np.ndarray
        Natural coordinates (ξ, η) in the reference square.
    Returns
    -------
    N : np.ndarray
        Shape functions evaluated at the input points. Shape: (n, 8, 1).
        Node order: [N1, N2, N3, N4, N5, N6, N7, N8].
    dN_dxi : np.ndarray
        Partial derivatives w.r.t. (ξ, η). Shape: (n, 8, 2).
        Columns correspond to [∂()/∂ξ, ∂()/∂η] in the same node order.
    Raises
    ------
    ValueError
        If `xi` is not a NumPy array.
        If `xi` has shape other than (2,) or (n, 2).
        If `xi` contains non-finite values (NaN or Inf).
    Notes
    -----
    Shape functions:
        N1 = -1/4 (1-ξ)(1-η)(1+ξ+η)
        N2 =  +1/4 (1+ξ)(1-η)(ξ-η-1)
        N3 =  +1/4 (1+ξ)(1+η)(ξ+η-1)
        N4 =  +1/4 (1-ξ)(1+η)(η-ξ-1)
        N5 =  +1/2 (1-ξ²)(1-η)
        N6 =  +1/2 (1+ξ)(1-η²)
        N7 =  +1/2 (1-ξ²)(1+η)
        N8 =  +1/2 (1-ξ)(1-η²)
    """
    if not isinstance(xi, np.ndarray):
        raise ValueError('`xi` must be a NumPy array.')
    if not np.all(np.isfinite(xi)):
        raise ValueError('`xi` contains non-finite values.')
    if xi.ndim == 1 and xi.shape == (2,):
        X = xi.reshape(1, 2).astype(float)
    elif xi.ndim == 2 and xi.shape[1] == 2:
        X = xi.astype(float, copy=False)
    else:
        raise ValueError('`xi` must have shape (2,) or (n, 2).')
    s = X[:, 0]
    t = X[:, 1]
    one = 1.0
    N1 = -0.25 * (one - s) * (one - t) * (one + s + t)
    N2 = 0.25 * (one + s) * (one - t) * (s - t - one)
    N3 = 0.25 * (one + s) * (one + t) * (s + t - one)
    N4 = 0.25 * (one - s) * (one + t) * (t - s - one)
    N5 = 0.5 * (one - s ** 2) * (one - t)
    N6 = 0.5 * (one + s) * (one - t ** 2)
    N7 = 0.5 * (one - s ** 2) * (one + t)
    N8 = 0.5 * (one - s) * (one - t ** 2)
    N = np.stack((N1, N2, N3, N4, N5, N6, N7, N8), axis=1)[..., np.newaxis]
    dN1_ds = 0.25 * (one - t) * (2.0 * s + t)
    dN1_dt = 0.25 * (one - s) * (s + 2.0 * t)
    dN2_ds = 0.25 * (one - t) * (2.0 * s - t)
    dN2_dt = 0.25 * (one + s) * (2.0 * t - s)
    dN3_ds = 0.25 * (one + t) * (2.0 * s + t)
    dN3_dt = 0.25 * (one + s) * (s + 2.0 * t)
    dN4_ds = 0.25 * (one + t) * (2.0 * s - t)
    dN4_dt = 0.25 * (one - s) * (2.0 * t - s)
    dN5_ds = -s * (one - t)
    dN5_dt = -0.5 * (one - s ** 2)
    dN6_ds = 0.5 * (one - t ** 2)
    dN6_dt = -(one + s) * t
    dN7_ds = -s * (one + t)
    dN7_dt = 0.5 * (one - s ** 2)
    dN8_ds = -0.5 * (one - t ** 2)
    dN8_dt = -(one - s) * t
    dN_ds = np.stack((dN1_ds, dN2_ds, dN3_ds, dN4_ds, dN5_ds, dN6_ds, dN7_ds, dN8_ds), axis=1)
    dN_dt = np.stack((dN1_dt, dN2_dt, dN3_dt, dN4_dt, dN5_dt, dN6_dt, dN7_dt, dN8_dt), axis=1)
    dN_dxi = np.stack((dN_ds, dN_dt), axis=2)
    return (N, dN_dxi)
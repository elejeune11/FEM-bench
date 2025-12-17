def FEM_2D_quad8_shape_fcns_and_derivatives_CC0_H0_T0(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        raise ValueError('xi must be a NumPy array.')
    if xi.ndim == 1:
        if xi.shape != (2,):
            raise ValueError('xi must have shape (2,) or (n, 2).')
        X = xi.reshape(1, 2)
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('xi must have shape (2,) or (n, 2).')
        X = xi
    else:
        raise ValueError('xi must have shape (2,) or (n, 2).')
    try:
        X = X.astype(float, copy=False)
    except Exception:
        raise ValueError('xi contains non-finite values (NaN or Inf).')
    if not np.isfinite(X).all():
        raise ValueError('xi contains non-finite values (NaN or Inf).')
    s = X[:, 0]
    t = X[:, 1]
    s2 = s * s
    t2 = t * t
    N1 = -0.25 * (1 - s) * (1 - t) * (1 + s + t)
    N2 = 0.25 * (1 + s) * (1 - t) * (s - t - 1)
    N3 = 0.25 * (1 + s) * (1 + t) * (s + t - 1)
    N4 = 0.25 * (1 - s) * (1 + t) * (t - s - 1)
    N5 = 0.5 * (1 - s2) * (1 - t)
    N6 = 0.5 * (1 + s) * (1 - t2)
    N7 = 0.5 * (1 - s2) * (1 + t)
    N8 = 0.5 * (1 - s) * (1 - t2)
    N = np.stack([N1, N2, N3, N4, N5, N6, N7, N8], axis=1)[:, :, np.newaxis]
    dNds1 = 0.25 * (1 - t) * (2 * s + t)
    dNds2 = 0.25 * (1 - t) * (2 * s - t)
    dNds3 = 0.25 * (1 + t) * (2 * s + t)
    dNds4 = 0.25 * (1 + t) * (2 * s - t)
    dNds5 = -s * (1 - t)
    dNds6 = 0.5 * (1 - t2)
    dNds7 = -s * (1 + t)
    dNds8 = -0.5 * (1 - t2)
    dNdt1 = 0.25 * (1 - s) * (s + 2 * t)
    dNdt2 = -0.25 * (1 + s) * (s - 2 * t)
    dNdt3 = 0.25 * (1 + s) * (s + 2 * t)
    dNdt4 = 0.25 * (1 - s) * (-s + 2 * t)
    dNdt5 = -0.5 * (1 - s2)
    dNdt6 = -(1 + s) * t
    dNdt7 = 0.5 * (1 - s2)
    dNdt8 = -(1 - s) * t
    dNds = np.stack([dNds1, dNds2, dNds3, dNds4, dNds5, dNds6, dNds7, dNds8], axis=1)
    dNdt = np.stack([dNdt1, dNdt2, dNdt3, dNdt4, dNdt5, dNdt6, dNdt7, dNdt8], axis=1)
    dN_dxi = np.stack([dNds, dNdt], axis=2)
    return (N, dN_dxi)
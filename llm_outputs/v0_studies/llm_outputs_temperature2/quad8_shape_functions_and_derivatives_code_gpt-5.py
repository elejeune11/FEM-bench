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
    if xi.ndim == 1:
        if xi.shape != (2,):
            raise ValueError('`xi` must have shape (2,) for a single point or (n, 2) for multiple points.')
        X = xi[None, :]
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('`xi` must have shape (2,) for a single point or (n, 2) for multiple points.')
        X = xi
    else:
        raise ValueError('`xi` must have shape (2,) for a single point or (n, 2) for multiple points.')
    if not np.all(np.isfinite(X)):
        raise ValueError('`xi` contains non-finite values (NaN or Inf).')
    s = X[:, 0]
    t = X[:, 1]
    N1 = -0.25 * (1.0 - s) * (1.0 - t) * (1.0 + s + t)
    N2 = 0.25 * (1.0 + s) * (1.0 - t) * (s - t - 1.0)
    N3 = 0.25 * (1.0 + s) * (1.0 + t) * (s + t - 1.0)
    N4 = 0.25 * (1.0 - s) * (1.0 + t) * (t - s - 1.0)
    N5 = 0.5 * (1.0 - s ** 2) * (1.0 - t)
    N6 = 0.5 * (1.0 + s) * (1.0 - t ** 2)
    N7 = 0.5 * (1.0 - s ** 2) * (1.0 + t)
    N8 = 0.5 * (1.0 - s) * (1.0 - t ** 2)
    dNds1 = 0.25 * (1.0 - t) * (2.0 * s + t)
    dNdt1 = 0.25 * (1.0 - s) * (s + 2.0 * t)
    dNds2 = 0.25 * (1.0 - t) * (2.0 * s - t)
    dNdt2 = -0.25 * (1.0 + s) * (s - 2.0 * t)
    dNds3 = 0.25 * (1.0 + t) * (2.0 * s + t)
    dNdt3 = 0.25 * (1.0 + s) * (s + 2.0 * t)
    dNds4 = -0.25 * (1.0 + t) * (t - 2.0 * s)
    dNdt4 = 0.25 * (1.0 - s) * (2.0 * t - s)
    dNds5 = -s * (1.0 - t)
    dNdt5 = -0.5 * (1.0 - s ** 2)
    dNds6 = 0.5 * (1.0 - t ** 2)
    dNdt6 = -(1.0 + s) * t
    dNds7 = -s * (1.0 + t)
    dNdt7 = 0.5 * (1.0 - s ** 2)
    dNds8 = -0.5 * (1.0 - t ** 2)
    dNdt8 = -(1.0 - s) * t
    N = np.stack([N1, N2, N3, N4, N5, N6, N7, N8], axis=1)[..., None]
    dNds = np.stack([dNds1, dNds2, dNds3, dNds4, dNds5, dNds6, dNds7, dNds8], axis=1)
    dNdt = np.stack([dNdt1, dNdt2, dNdt3, dNdt4, dNdt5, dNdt6, dNdt7, dNdt8], axis=1)
    dN_dxi = np.stack([dNds, dNdt], axis=2)
    return (N, dN_dxi)
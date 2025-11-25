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
        if xi.shape == (2,):
            xi = xi.reshape(1, 2)
        else:
            raise ValueError('`xi` must have shape (2,) or (n, 2).')
    elif xi.ndim != 2 or xi.shape[1] != 2:
        raise ValueError('`xi` must have shape (2,) or (n, 2).')
    if not np.all(np.isfinite(xi)):
        raise ValueError('`xi` must contain finite values.')
    x = xi[:, 0]
    y = xi[:, 1]
    N1 = -0.25 * (1 - x) * (1 - y) * (1 + x + y)
    N2 = 0.25 * (1 + x) * (1 - y) * (x - y - 1)
    N3 = 0.25 * (1 + x) * (1 + y) * (x + y - 1)
    N4 = 0.25 * (1 - x) * (1 + y) * (y - x - 1)
    N5 = 0.5 * (1 - x ** 2) * (1 - y)
    N6 = 0.5 * (1 + x) * (1 - y ** 2)
    N7 = 0.5 * (1 - x ** 2) * (1 + y)
    N8 = 0.5 * (1 - x) * (1 - y ** 2)
    N = np.stack([N1, N2, N3, N4, N5, N6, N7, N8], axis=1)
    N = N[..., np.newaxis]
    dN1_dx = 0.25 * (1 - y) * (2 * x + y)
    dN2_dx = 0.25 * (1 - y) * (2 * x - y)
    dN3_dx = 0.25 * (1 + y) * (2 * x + y)
    dN4_dx = 0.25 * (1 + y) * (2 * x - y)
    dN5_dx = -x * (1 - y)
    dN6_dx = 0.5 * (1 - y ** 2)
    dN7_dx = -x * (1 + y)
    dN8_dx = -0.5 * (1 - y ** 2)
    dN1_dy = 0.25 * (1 - x) * (x + 2 * y)
    dN2_dy = 0.25 * (1 + x) * (-x + 2 * y)
    dN3_dy = 0.25 * (1 + x) * (x + 2 * y)
    dN4_dy = 0.25 * (1 - x) * (-x + 2 * y)
    dN5_dy = -0.5 * (1 - x ** 2)
    dN6_dy = -y * (1 + x)
    dN7_dy = 0.5 * (1 - x ** 2)
    dN8_dy = -y * (1 - x)
    dN_dx = np.stack([dN1_dx, dN2_dx, dN3_dx, dN4_dx, dN5_dx, dN6_dx, dN7_dx, dN8_dx], axis=1)
    dN_dy = np.stack([dN1_dy, dN2_dy, dN3_dy, dN4_dy, dN5_dy, dN6_dy, dN7_dy, dN8_dy], axis=1)
    dN_dxi = np.stack([dN_dx, dN_dy], axis=2)
    return (N, dN_dxi)
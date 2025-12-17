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
        if xi.shape[0] != 2:
            raise ValueError('xi must have shape (2,) or (n, 2).')
        if not np.all(np.isfinite(xi)):
            raise ValueError('xi contains non-finite values.')
        x = np.array([xi[0]], dtype=np.float64)
        y = np.array([xi[1]], dtype=np.float64)
        n = 1
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('xi must have shape (2,) or (n, 2).')
        if not np.all(np.isfinite(xi)):
            raise ValueError('xi contains non-finite values.')
        n = xi.shape[0]
        x = xi[:, 0].astype(np.float64, copy=False)
        y = xi[:, 1].astype(np.float64, copy=False)
    else:
        raise ValueError('xi must have shape (2,) or (n, 2).')
    one = 1.0
    two = 2.0
    half = 0.5
    quarter = 0.25
    N1 = -quarter * (one - x) * (one - y) * (one + x + y)
    N2 = quarter * (one + x) * (one - y) * (x - y - one)
    N3 = quarter * (one + x) * (one + y) * (x + y - one)
    N4 = quarter * (one - x) * (one + y) * (y - x - one)
    N5 = half * (one - x * x) * (one - y)
    N6 = half * (one + x) * (one - y * y)
    N7 = half * (one - x * x) * (one + y)
    N8 = half * (one - x) * (one - y * y)
    dN1_dx = quarter * (one - y) * (two * x + y)
    dN2_dx = quarter * (one - y) * (two * x - y)
    dN3_dx = quarter * (one + y) * (two * x + y)
    dN4_dx = quarter * (one + y) * (two * x - y)
    dN5_dx = -x * (one - y)
    dN6_dx = half * (one - y * y)
    dN7_dx = -x * (one + y)
    dN8_dx = -half * (one - y * y)
    dN1_dy = quarter * (one - x) * (x + two * y)
    dN2_dy = -quarter * (one + x) * (x - two * y)
    dN3_dy = quarter * (one + x) * (x + two * y)
    dN4_dy = quarter * (one - x) * (two * y - x)
    dN5_dy = -half * (one - x * x)
    dN6_dy = -(one + x) * y
    dN7_dy = half * (one - x * x)
    dN8_dy = -(one - x) * y
    N = np.stack([N1, N2, N3, N4, N5, N6, N7, N8], axis=1).reshape(n, 8, 1)
    dN_dx = np.stack([dN1_dx, dN2_dx, dN3_dx, dN4_dx, dN5_dx, dN6_dx, dN7_dx, dN8_dx], axis=1)
    dN_dy = np.stack([dN1_dy, dN2_dy, dN3_dy, dN4_dy, dN5_dy, dN6_dy, dN7_dy, dN8_dy], axis=1)
    dN_dxi = np.stack([dN_dx, dN_dy], axis=2)
    return (N, dN_dxi)
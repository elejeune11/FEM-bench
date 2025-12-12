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
        raise ValueError('`xi` must be a NumPy array.')
    if xi.ndim == 1:
        if xi.shape != (2,):
            raise ValueError('`xi` must have shape (2,) for a single point or (n, 2) for multiple points.')
        xi_arr = xi.reshape(1, 2)
    elif xi.ndim == 2 and xi.shape[1] == 2:
        xi_arr = xi
    else:
        raise ValueError('`xi` must have shape (2,) for a single point or (n, 2) for multiple points.')
    if not np.all(np.isfinite(xi_arr)):
        raise ValueError('`xi` contains non-finite values.')
    x = xi_arr[:, 0]
    y = xi_arr[:, 1]
    one = 1.0
    N1 = -0.25 * (one - x) * (one - y) * (one + x + y)
    N2 = 0.25 * (one + x) * (one - y) * (x - y - one)
    N3 = 0.25 * (one + x) * (one + y) * (x + y - one)
    N4 = 0.25 * (one - x) * (one + y) * (y - x - one)
    N5 = 0.5 * (one - x * x) * (one - y)
    N6 = 0.5 * (one + x) * (one - y * y)
    N7 = 0.5 * (one - x * x) * (one + y)
    N8 = 0.5 * (one - x) * (one - y * y)
    dN1_dx = 0.25 * (one - y) * (2.0 * x + y)
    dN1_dy = 0.25 * (one - x) * (x + 2.0 * y)
    dN2_dx = 0.25 * (one - y) * (2.0 * x - y)
    dN2_dy = -0.25 * (one + x) * (x - 2.0 * y)
    dN3_dx = 0.25 * (one + y) * (2.0 * x + y)
    dN3_dy = 0.25 * (one + x) * (x + 2.0 * y)
    dN4_dx = 0.25 * (one + y) * (2.0 * x - y)
    dN4_dy = 0.25 * (one - x) * (2.0 * y - x)
    dN5_dx = -x * (one - y)
    dN5_dy = -0.5 * (one - x * x)
    dN6_dx = 0.5 * (one - y * y)
    dN6_dy = -y * (one + x)
    dN7_dx = -x * (one + y)
    dN7_dy = 0.5 * (one - x * x)
    dN8_dx = -0.5 * (one - y * y)
    dN8_dy = -y * (one - x)
    N = np.stack([N1, N2, N3, N4, N5, N6, N7, N8], axis=1)[..., None]
    dN_dxi_x = np.stack([dN1_dx, dN2_dx, dN3_dx, dN4_dx, dN5_dx, dN6_dx, dN7_dx, dN8_dx], axis=1)
    dN_dxi_y = np.stack([dN1_dy, dN2_dy, dN3_dy, dN4_dy, dN5_dy, dN6_dy, dN7_dy, dN8_dy], axis=1)
    dN_dxi = np.stack([dN_dxi_x, dN_dxi_y], axis=2)
    return (N, dN_dxi)
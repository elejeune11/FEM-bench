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
        raise ValueError('Input `xi` must be a NumPy array.')
    if xi.ndim == 1:
        if xi.shape[0] != 2:
            raise ValueError('Input `xi` must have shape (2,) or (n, 2).')
        xi = xi[np.newaxis, :]
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('Input `xi` must have shape (2,) or (n, 2).')
    else:
        raise ValueError('Input `xi` must have shape (2,) or (n, 2).')
    if not np.all(np.isfinite(xi)):
        raise ValueError('Input `xi` contains non-finite values.')
    n_points = xi.shape[0]
    (xi_val, eta_val) = (xi[:, 0], xi[:, 1])
    N1 = -0.25 * (1 - xi_val) * (1 - eta_val) * (1 + xi_val + eta_val)
    N2 = 0.25 * (1 + xi_val) * (1 - eta_val) * (xi_val - eta_val - 1)
    N3 = 0.25 * (1 + xi_val) * (1 + eta_val) * (xi_val + eta_val - 1)
    N4 = 0.25 * (1 - xi_val) * (1 + eta_val) * (eta_val - xi_val - 1)
    N5 = 0.5 * (1 - xi_val ** 2) * (1 - eta_val)
    N6 = 0.5 * (1 + xi_val) * (1 - eta_val ** 2)
    N7 = 0.5 * (1 - xi_val ** 2) * (1 + eta_val)
    N8 = 0.5 * (1 - xi_val) * (1 - eta_val ** 2)
    N = np.stack([N1, N2, N3, N4, N5, N6, N7, N8], axis=1)[:, :, np.newaxis]
    dN1_dxi = 0.25 * (1 - eta_val) * (2 * xi_val + eta_val)
    dN1_deta = 0.25 * (1 - xi_val) * (xi_val + 2 * eta_val)
    dN2_dxi = 0.25 * (1 - eta_val) * (2 * xi_val - eta_val - 1)
    dN2_deta = 0.25 * (1 + xi_val) * (-xi_val - 2 * eta_val + 1)
    dN3_dxi = 0.25 * (1 + eta_val) * (2 * xi_val + eta_val - 1)
    dN3_deta = 0.25 * (1 + xi_val) * (xi_val + 2 * eta_val - 1)
    dN4_dxi = 0.25 * (1 + eta_val) * (-2 * xi_val + eta_val + 1)
    dN4_deta = 0.25 * (1 - xi_val) * (2 * eta_val - xi_val - 1)
    dN5_dxi = -xi_val * (1 - eta_val)
    dN5_deta = -0.5 * (1 - xi_val ** 2)
    dN6_dxi = 0.5 * (1 - eta_val ** 2)
    dN6_deta = -(1 + xi_val) * eta_val
    dN7_dxi = -xi_val * (1 + eta_val)
    dN7_deta = 0.5 * (1 - xi_val ** 2)
    dN8_dxi = -0.5 * (1 - eta_val ** 2)
    dN8_deta = -(1 - xi_val) * eta_val
    dN_dxi = np.stack([np.stack([dN1_dxi, dN1_deta], axis=1), np.stack([dN2_dxi, dN2_deta], axis=1), np.stack([dN3_dxi, dN3_deta], axis=1), np.stack([dN4_dxi, dN4_deta], axis=1), np.stack([dN5_dxi, dN5_deta], axis=1), np.stack([dN6_dxi, dN6_deta], axis=1), np.stack([dN7_dxi, dN7_deta], axis=1), np.stack([dN8_dxi, dN8_deta], axis=1)], axis=1)
    return (N, dN_dxi)
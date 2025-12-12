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
        raise ValueError('xi must be a NumPy array')
    if xi.ndim == 1:
        if xi.shape != (2,):
            raise ValueError('xi must have shape (2,) or (n, 2)')
        xi_arr = xi.reshape(1, 2)
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('xi must have shape (2,) or (n, 2)')
        xi_arr = xi
    else:
        raise ValueError('xi must have shape (2,) or (n, 2)')
    if not np.isfinite(xi_arr).all():
        raise ValueError('xi contains non-finite values')
    xi0 = xi_arr[:, 0].astype(float)
    eta = xi_arr[:, 1].astype(float)
    one_minus_xi = 1.0 - xi0
    one_plus_xi = 1.0 + xi0
    one_minus_eta = 1.0 - eta
    one_plus_eta = 1.0 + eta
    N1 = -0.25 * one_minus_xi * one_minus_eta * (1.0 + xi0 + eta)
    N2 = 0.25 * one_plus_xi * one_minus_eta * (xi0 - eta - 1.0)
    N3 = 0.25 * one_plus_xi * one_plus_eta * (xi0 + eta - 1.0)
    N4 = 0.25 * one_minus_xi * one_plus_eta * (eta - xi0 - 1.0)
    N5 = 0.5 * (1.0 - xi0 ** 2) * one_minus_eta
    N6 = 0.5 * one_plus_xi * (1.0 - eta ** 2)
    N7 = 0.5 * (1.0 - xi0 ** 2) * one_plus_eta
    N8 = 0.5 * one_minus_xi * (1.0 - eta ** 2)
    N_mat = np.column_stack((N1, N2, N3, N4, N5, N6, N7, N8))
    N_out = N_mat[:, :, None]
    dN1_dxi = 0.25 * one_minus_eta * (2.0 * xi0 + eta)
    dN1_deta = 0.25 * one_minus_xi * (2.0 * eta + xi0)
    dN2_dxi = 0.25 * one_minus_eta * (2.0 * xi0 - eta)
    dN2_deta = -0.25 * one_plus_xi * (xi0 - 2.0 * eta)
    dN3_dxi = 0.25 * one_plus_eta * (2.0 * xi0 + eta)
    dN3_deta = 0.25 * one_plus_xi * (xi0 + 2.0 * eta)
    dN4_dxi = 0.25 * one_plus_eta * (2.0 * xi0 - eta)
    dN4_deta = 0.25 * one_minus_xi * (2.0 * eta - xi0)
    dN5_dxi = -xi0 * one_minus_eta
    dN5_deta = -0.5 * (1.0 - xi0 ** 2)
    dN6_dxi = 0.5 * (1.0 - eta ** 2)
    dN6_deta = -(1.0 + xi0) * eta
    dN7_dxi = -xi0 * one_plus_eta
    dN7_deta = 0.5 * (1.0 - xi0 ** 2)
    dN8_dxi = -0.5 * (1.0 - eta ** 2)
    dN8_deta = -(1.0 - xi0) * eta
    dxi_cols = np.column_stack((dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi, dN5_dxi, dN6_dxi, dN7_dxi, dN8_dxi))
    deta_cols = np.column_stack((dN1_deta, dN2_deta, dN3_deta, dN4_deta, dN5_deta, dN6_deta, dN7_deta, dN8_deta))
    dN_dxi_out = np.stack((dxi_cols, deta_cols), axis=2)
    return (N_out, dN_dxi_out)
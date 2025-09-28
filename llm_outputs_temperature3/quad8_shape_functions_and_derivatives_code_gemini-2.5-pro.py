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
        raise ValueError('`xi` is not a NumPy array.')
    if xi.ndim == 1:
        if xi.shape != (2,):
            raise ValueError('`xi` has shape other than (2,) or (n, 2).')
        xi = xi.reshape(1, 2)
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('`xi` has shape other than (2,) or (n, 2).')
    else:
        raise ValueError('`xi` has shape other than (2,) or (n, 2).')
    if not np.all(np.isfinite(xi)):
        raise ValueError('`xi` contains non-finite values (NaN or Inf).')
    xi_ = xi[:, 0]
    eta = xi[:, 1]
    N1 = -0.25 * (1 - xi_) * (1 - eta) * (1 + xi_ + eta)
    N2 = 0.25 * (1 + xi_) * (1 - eta) * (xi_ - eta - 1)
    N3 = 0.25 * (1 + xi_) * (1 + eta) * (xi_ + eta - 1)
    N4 = 0.25 * (1 - xi_) * (1 + eta) * (eta - xi_ - 1)
    N5 = 0.5 * (1 - xi_ ** 2) * (1 - eta)
    N6 = 0.5 * (1 + xi_) * (1 - eta ** 2)
    N7 = 0.5 * (1 - xi_ ** 2) * (1 + eta)
    N8 = 0.5 * (1 - xi_) * (1 - eta ** 2)
    N = np.stack([N1, N2, N3, N4, N5, N6, N7, N8], axis=1)
    N = N[:, :, np.newaxis]
    dN1_dxi = 0.25 * (1 - eta) * (2 * xi_ + eta)
    dN2_dxi = 0.25 * (1 - eta) * (2 * xi_ - eta)
    dN3_dxi = 0.25 * (1 + eta) * (2 * xi_ + eta)
    dN4_dxi = 0.25 * (1 + eta) * (2 * xi_ - eta)
    dN5_dxi = -xi_ * (1 - eta)
    dN6_dxi = 0.5 * (1 - eta ** 2)
    dN7_dxi = -xi_ * (1 + eta)
    dN8_dxi = -0.5 * (1 - eta ** 2)
    dN1_deta = 0.25 * (1 - xi_) * (xi_ + 2 * eta)
    dN2_deta = 0.25 * (1 + xi_) * (-xi_ + 2 * eta)
    dN3_deta = 0.25 * (1 + xi_) * (xi_ + 2 * eta)
    dN4_deta = 0.25 * (1 - xi_) * (2 * eta - xi_)
    dN5_deta = -0.5 * (1 - xi_ ** 2)
    dN6_deta = -eta * (1 + xi_)
    dN7_deta = 0.5 * (1 - xi_ ** 2)
    dN8_deta = -eta * (1 - xi_)
    dN_dxi_comp = np.stack([dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi, dN5_dxi, dN6_dxi, dN7_dxi, dN8_dxi], axis=1)
    dN_deta_comp = np.stack([dN1_deta, dN2_deta, dN3_deta, dN4_deta, dN5_deta, dN6_deta, dN7_deta, dN8_deta], axis=1)
    dN_dxi = np.stack([dN_dxi_comp, dN_deta_comp], axis=2)
    return (N, dN_dxi)
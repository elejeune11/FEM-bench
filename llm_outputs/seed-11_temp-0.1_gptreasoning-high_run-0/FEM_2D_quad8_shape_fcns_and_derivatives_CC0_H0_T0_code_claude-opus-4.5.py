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
        if xi.shape[0] != 2:
            raise ValueError('xi must have shape (2,) or (n, 2)')
        xi = xi.reshape(1, 2)
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('xi must have shape (2,) or (n, 2)')
    else:
        raise ValueError('xi must have shape (2,) or (n, 2)')
    if not np.all(np.isfinite(xi)):
        raise ValueError('xi contains non-finite values (NaN or Inf)')
    n = xi.shape[0]
    xiv = xi[:, 0]
    eta = xi[:, 1]
    one_minus_xi = 1 - xiv
    one_plus_xi = 1 + xiv
    one_minus_eta = 1 - eta
    one_plus_eta = 1 + eta
    one_minus_xi_sq = 1 - xiv ** 2
    one_minus_eta_sq = 1 - eta ** 2
    N1 = -0.25 * one_minus_xi * one_minus_eta * (1 + xiv + eta)
    N2 = 0.25 * one_plus_xi * one_minus_eta * (xiv - eta - 1)
    N3 = 0.25 * one_plus_xi * one_plus_eta * (xiv + eta - 1)
    N4 = 0.25 * one_minus_xi * one_plus_eta * (eta - xiv - 1)
    N5 = 0.5 * one_minus_xi_sq * one_minus_eta
    N6 = 0.5 * one_plus_xi * one_minus_eta_sq
    N7 = 0.5 * one_minus_xi_sq * one_plus_eta
    N8 = 0.5 * one_minus_xi * one_minus_eta_sq
    N = np.stack([N1, N2, N3, N4, N5, N6, N7, N8], axis=1).reshape(n, 8, 1)
    dN1_dxi = -0.25 * (-one_minus_eta * (1 + xiv + eta) + one_minus_xi * one_minus_eta * -1)
    dN1_dxi = -0.25 * (-(1 - eta) * (1 + xiv + eta) + (1 - xiv) * (1 - eta) * -1)
    dN1_dxi = -0.25 * one_minus_eta * (-(1 + xiv + eta) - (1 - xiv))
    dN1_dxi = -0.25 * one_minus_eta * (-1 - xiv - eta - 1 + xiv)
    dN1_dxi = -0.25 * one_minus_eta * (-2 - eta)
    dN1_dxi = 0.25 * one_minus_eta * (2 * xiv + eta)
    dN2_dxi = 0.25 * one_minus_eta * (xiv - eta - 1 + one_plus_xi)
    dN2_dxi = 0.25 * one_minus_eta * (2 * xiv - eta)
    dN3_dxi = 0.25 * one_plus_eta * (xiv + eta - 1 + one_plus_xi)
    dN3_dxi = 0.25 * one_plus_eta * (2 * xiv + eta)
    dN4_dxi = 0.25 * one_plus_eta * (-(eta - xiv - 1) + one_minus_xi * -1)
    dN4_dxi = 0.25 * one_plus_eta * (-eta + xiv + 1 - 1 + xiv)
    dN4_dxi = 0.25 * one_plus_eta * (2 * xiv - eta)
    dN5_dxi = 0.5 * (-2 * xiv) * one_minus_eta
    dN5_dxi = -xiv * one_minus_eta
    dN6_dxi = 0.5 * one_minus_eta_sq
    dN7_dxi = 0.5 * (-2 * xiv) * one_plus_eta
    dN7_dxi = -xiv * one_plus_eta
    dN8_dxi = -0.5 * one_minus_eta_sq
    dN1_deta = 0.25 * one_minus_xi * (2 * eta + xiv)
    dN2_deta = 0.25 * one_plus_xi * (2 * eta - xiv)
    dN3_deta = 0.25 * one_plus_xi * (2 * eta + xiv)
    dN4_deta = 0.25 * one_minus_xi * (2 * eta - xiv)
    dN5_deta = -0.5 * one_minus_xi_sq
    dN6_deta = -eta * one_plus_xi
    dN7_deta = 0.5 * one_minus_xi_sq
    dN8_deta = -eta * one_minus_xi
    dN_dxi_col = np.stack([dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi, dN5_dxi, dN6_dxi, dN7_dxi, dN8_dxi], axis=1)
    dN_deta_col = np.stack([dN1_deta, dN2_deta, dN3_deta, dN4_deta, dN5_deta, dN6_deta, dN7_deta, dN8_deta], axis=1)
    dN_dxi = np.stack([dN_dxi_col, dN_deta_col], axis=2)
    return (N, dN_dxi)
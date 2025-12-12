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
    xi_vals = xi[:, 0]
    eta_vals = xi[:, 1]
    N = np.zeros((n, 8, 1))
    dN_dxi = np.zeros((n, 8, 2))
    N1 = -0.25 * (1 - xi_vals) * (1 - eta_vals) * (1 + xi_vals + eta_vals)
    N2 = 0.25 * (1 + xi_vals) * (1 - eta_vals) * (xi_vals - eta_vals - 1)
    N3 = 0.25 * (1 + xi_vals) * (1 + eta_vals) * (xi_vals + eta_vals - 1)
    N4 = 0.25 * (1 - xi_vals) * (1 + eta_vals) * (eta_vals - xi_vals - 1)
    N5 = 0.5 * (1 - xi_vals ** 2) * (1 - eta_vals)
    N6 = 0.5 * (1 + xi_vals) * (1 - eta_vals ** 2)
    N7 = 0.5 * (1 - xi_vals ** 2) * (1 + eta_vals)
    N8 = 0.5 * (1 - xi_vals) * (1 - eta_vals ** 2)
    N[:, 0, 0] = N1
    N[:, 1, 0] = N2
    N[:, 2, 0] = N3
    N[:, 3, 0] = N4
    N[:, 4, 0] = N5
    N[:, 5, 0] = N6
    N[:, 6, 0] = N7
    N[:, 7, 0] = N8
    dN1_dxi = -0.25 * (-(1 - eta_vals) * (1 + xi_vals + eta_vals) + (1 - xi_vals) * (1 - eta_vals))
    dN1_deta = -0.25 * (-(1 - xi_vals) * (1 + xi_vals + eta_vals) + (1 - xi_vals) * (1 + xi_vals))
    dN2_dxi = 0.25 * ((1 - eta_vals) * (xi_vals - eta_vals - 1) + (1 + xi_vals) * (1 - eta_vals))
    dN2_deta = 0.25 * (-(1 + xi_vals) * (xi_vals - eta_vals - 1) + (1 + xi_vals) * (1 - eta_vals) * -1)
    dN3_dxi = 0.25 * ((1 + eta_vals) * (xi_vals + eta_vals - 1) + (1 + xi_vals) * (1 + eta_vals))
    dN3_deta = 0.25 * ((1 + xi_vals) * (xi_vals + eta_vals - 1) + (1 + xi_vals) * (1 + eta_vals))
    dN4_dxi = 0.25 * (-(1 + eta_vals) * (eta_vals - xi_vals - 1) + (1 - xi_vals) * (1 + eta_vals))
    dN4_deta = 0.25 * ((1 - xi_vals) * (eta_vals - xi_vals - 1) + (1 - xi_vals) * (1 + eta_vals))
    dN5_dxi = 0.5 * (-2 * xi_vals) * (1 - eta_vals)
    dN5_deta = 0.5 * (1 - xi_vals ** 2) * -1
    dN6_dxi = 0.5 * (1 - eta_vals ** 2)
    dN6_deta = 0.5 * (1 + xi_vals) * (-2 * eta_vals)
    dN7_dxi = 0.5 * (-2 * xi_vals) * (1 + eta_vals)
    dN7_deta = 0.5 * (1 - xi_vals ** 2)
    dN8_dxi = 0.5 * -1 * (1 - eta_vals ** 2)
    dN8_deta = 0.5 * (1 - xi_vals) * (-2 * eta_vals)
    dN_dxi[:, 0, 0] = dN1_dxi
    dN_dxi[:, 0, 1] = dN1_deta
    dN_dxi[:, 1, 0] = dN2_dxi
    dN_dxi[:, 1, 1] = dN2_deta
    dN_dxi[:, 2, 0] = dN3_dxi
    dN_dxi[:, 2, 1] = dN3_deta
    dN_dxi[:, 3, 0] = dN4_dxi
    dN_dxi[:, 3, 1] = dN4_deta
    dN_dxi[:, 4, 0] = dN5_dxi
    dN_dxi[:, 4, 1] = dN5_deta
    dN_dxi[:, 5, 0] = dN6_dxi
    dN_dxi[:, 5, 1] = dN6_deta
    dN_dxi[:, 6, 0] = dN7_dxi
    dN_dxi[:, 6, 1] = dN7_deta
    dN_dxi[:, 7, 0] = dN8_dxi
    dN_dxi[:, 7, 1] = dN8_deta
    return (N, dN_dxi)
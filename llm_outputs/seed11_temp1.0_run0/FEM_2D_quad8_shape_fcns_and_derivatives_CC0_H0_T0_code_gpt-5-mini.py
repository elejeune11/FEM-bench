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
            raise ValueError('xi must have shape (2,) or (n,2)')
        xi_arr = xi.reshape(1, 2).astype(float)
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('xi must have shape (2,) or (n,2)')
        xi_arr = xi.astype(float)
    else:
        raise ValueError('xi must have shape (2,) or (n,2)')
    if not np.isfinite(xi_arr).all():
        raise ValueError('xi contains non-finite values')
    xi1 = xi_arr[:, 0]
    eta = xi_arr[:, 1]
    one_minus_xi = 1.0 - xi1
    one_plus_xi = 1.0 + xi1
    one_minus_eta = 1.0 - eta
    one_plus_eta = 1.0 + eta
    one_minus_xi2 = 1.0 - xi1 * xi1
    one_minus_eta2 = 1.0 - eta * eta
    N1 = -0.25 * one_minus_xi * one_minus_eta * (1.0 + xi1 + eta)
    N2 = 0.25 * one_plus_xi * one_minus_eta * (xi1 - eta - 1.0)
    N3 = 0.25 * one_plus_xi * one_plus_eta * (xi1 + eta - 1.0)
    N4 = 0.25 * one_minus_xi * one_plus_eta * (eta - xi1 - 1.0)
    N5 = 0.5 * one_minus_xi2 * one_minus_eta
    N6 = 0.5 * one_plus_xi * one_minus_eta2
    N7 = 0.5 * one_minus_xi2 * one_plus_eta
    N8 = 0.5 * one_minus_xi * one_minus_eta2
    n = xi_arr.shape[0]
    N = np.zeros((n, 8, 1), dtype=float)
    N[:, 0, 0] = N1
    N[:, 1, 0] = N2
    N[:, 2, 0] = N3
    N[:, 3, 0] = N4
    N[:, 4, 0] = N5
    N[:, 5, 0] = N6
    N[:, 6, 0] = N7
    N[:, 7, 0] = N8
    dN = np.zeros((n, 8, 2), dtype=float)
    a = one_minus_xi
    b = one_minus_eta
    c = 1.0 + xi1 + eta
    dN[:, 0, 0] = -0.25 * b * (a - c)
    dN[:, 0, 1] = -0.25 * a * (b - c)
    a = one_plus_xi
    b = one_minus_eta
    c = xi1 - eta - 1.0
    dN[:, 1, 0] = 0.25 * b * (c + a)
    dN[:, 1, 1] = -0.25 * a * (c + b)
    a = one_plus_xi
    b = one_plus_eta
    c = xi1 + eta - 1.0
    dN[:, 2, 0] = 0.25 * b * (c + a)
    dN[:, 2, 1] = 0.25 * a * (c + b)
    a = one_minus_xi
    b = one_plus_eta
    c = eta - xi1 - 1.0
    dN[:, 3, 0] = -0.25 * b * (c + a)
    dN[:, 3, 1] = 0.25 * a * (c + b)
    dN[:, 4, 0] = -xi1 * one_minus_eta
    dN[:, 4, 1] = -0.5 * one_minus_xi2
    dN[:, 5, 0] = 0.5 * one_minus_eta2
    dN[:, 5, 1] = -one_plus_xi * eta
    dN[:, 6, 0] = -xi1 * one_plus_eta
    dN[:, 6, 1] = 0.5 * one_minus_xi2
    dN[:, 7, 0] = -0.5 * one_minus_eta2
    dN[:, 7, 1] = -xi1 * eta + eta * xi1 * 0.0
    dN[:, 7, 1] = -(1.0 - xi1) * eta
    return (N, dN)
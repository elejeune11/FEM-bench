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
        raise ValueError('xi contains non-finite values')
    n_points = xi.shape[0]
    xi_vals = xi[:, 0]
    eta_vals = xi[:, 1]
    N = np.zeros((n_points, 8, 1))
    dN_dxi = np.zeros((n_points, 8, 2))
    xi_sq = xi_vals ** 2
    eta_sq = eta_vals ** 2
    one_minus_xi = 1 - xi_vals
    one_plus_xi = 1 + xi_vals
    one_minus_eta = 1 - eta_vals
    one_plus_eta = 1 + eta_vals
    N[:, 0, 0] = -0.25 * one_minus_xi * one_minus_eta * (1 + xi_vals + eta_vals)
    dN_dxi[:, 0, 0] = -0.25 * (-one_minus_eta * (1 + xi_vals + eta_vals) + one_minus_xi * one_minus_eta)
    dN_dxi[:, 0, 1] = -0.25 * (-one_minus_xi * (1 + xi_vals + eta_vals) + one_minus_xi * one_minus_eta)
    N[:, 1, 0] = 0.25 * one_plus_xi * one_minus_eta * (xi_vals - eta_vals - 1)
    dN_dxi[:, 1, 0] = 0.25 * (one_minus_eta * (xi_vals - eta_vals - 1) + one_plus_xi * one_minus_eta)
    dN_dxi[:, 1, 1] = 0.25 * (-one_plus_xi * (xi_vals - eta_vals - 1) - one_plus_xi * one_minus_eta)
    N[:, 2, 0] = 0.25 * one_plus_xi * one_plus_eta * (xi_vals + eta_vals - 1)
    dN_dxi[:, 2, 0] = 0.25 * (one_plus_eta * (xi_vals + eta_vals - 1) + one_plus_xi * one_plus_eta)
    dN_dxi[:, 2, 1] = 0.25 * (one_plus_xi * (xi_vals + eta_vals - 1) + one_plus_xi * one_plus_eta)
    N[:, 3, 0] = 0.25 * one_minus_xi * one_plus_eta * (eta_vals - xi_vals - 1)
    dN_dxi[:, 3, 0] = 0.25 * (-one_plus_eta * (eta_vals - xi_vals - 1) - one_minus_xi * one_plus_eta)
    dN_dxi[:, 3, 1] = 0.25 * (one_minus_xi * (eta_vals - xi_vals - 1) + one_minus_xi * one_plus_eta)
    N[:, 4, 0] = 0.5 * (1 - xi_sq) * one_minus_eta
    dN_dxi[:, 4, 0] = 0.5 * (-2 * xi_vals) * one_minus_eta
    dN_dxi[:, 4, 1] = 0.5 * (1 - xi_sq) * -1
    N[:, 5, 0] = 0.5 * one_plus_xi * (1 - eta_sq)
    dN_dxi[:, 5, 0] = 0.5 * (1 - eta_sq)
    dN_dxi[:, 5, 1] = 0.5 * one_plus_xi * (-2 * eta_vals)
    N[:, 6, 0] = 0.5 * (1 - xi_sq) * one_plus_eta
    dN_dxi[:, 6, 0] = 0.5 * (-2 * xi_vals) * one_plus_eta
    dN_dxi[:, 6, 1] = 0.5 * (1 - xi_sq)
    N[:, 7, 0] = 0.5 * one_minus_xi * (1 - eta_sq)
    dN_dxi[:, 7, 0] = 0.5 * -1 * (1 - eta_sq)
    dN_dxi[:, 7, 1] = 0.5 * one_minus_xi * (-2 * eta_vals)
    return (N, dN_dxi)
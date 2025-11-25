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
    n = xi.shape[0]
    xi_val = xi[:, 0]
    eta_val = xi[:, 1]
    N = np.zeros((n, 8, 1))
    dN_dxi = np.zeros((n, 8, 2))
    N[:, 0, 0] = -0.25 * (1 - xi_val) * (1 - eta_val) * (1 + xi_val + eta_val)
    N[:, 1, 0] = 0.25 * (1 + xi_val) * (1 - eta_val) * (xi_val - eta_val - 1)
    N[:, 2, 0] = 0.25 * (1 + xi_val) * (1 + eta_val) * (xi_val + eta_val - 1)
    N[:, 3, 0] = 0.25 * (1 - xi_val) * (1 + eta_val) * (eta_val - xi_val - 1)
    N[:, 4, 0] = 0.5 * (1 - xi_val ** 2) * (1 - eta_val)
    N[:, 5, 0] = 0.5 * (1 + xi_val) * (1 - eta_val ** 2)
    N[:, 6, 0] = 0.5 * (1 - xi_val ** 2) * (1 + eta_val)
    N[:, 7, 0] = 0.5 * (1 - xi_val) * (1 - eta_val ** 2)
    dN_dxi[:, 0, 0] = 0.25 * (1 - eta_val) * (2 * xi_val + eta_val)
    dN_dxi[:, 1, 0] = 0.25 * (1 - eta_val) * (2 * xi_val - eta_val)
    dN_dxi[:, 2, 0] = 0.25 * (1 + eta_val) * (2 * xi_val + eta_val)
    dN_dxi[:, 3, 0] = 0.25 * (1 + eta_val) * (2 * xi_val - eta_val)
    dN_dxi[:, 4, 0] = -xi_val * (1 - eta_val)
    dN_dxi[:, 5, 0] = 0.5 * (1 - eta_val ** 2)
    dN_dxi[:, 6, 0] = -xi_val * (1 + eta_val)
    dN_dxi[:, 7, 0] = -0.5 * (1 - eta_val ** 2)
    dN_dxi[:, 0, 1] = 0.25 * (1 - xi_val) * (xi_val + 2 * eta_val)
    dN_dxi[:, 1, 1] = 0.25 * (1 + xi_val) * (-xi_val + 2 * eta_val)
    dN_dxi[:, 2, 1] = 0.25 * (1 + xi_val) * (xi_val + 2 * eta_val)
    dN_dxi[:, 3, 1] = 0.25 * (1 - xi_val) * (-xi_val + 2 * eta_val)
    dN_dxi[:, 4, 1] = -0.5 * (1 - xi_val ** 2)
    dN_dxi[:, 5, 1] = -eta_val * (1 + xi_val)
    dN_dxi[:, 6, 1] = 0.5 * (1 - xi_val ** 2)
    dN_dxi[:, 7, 1] = -eta_val * (1 - xi_val)
    return (N, dN_dxi)
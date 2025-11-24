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
        raise ValueError('`xi` must be a NumPy array.')
    if not np.all(np.isfinite(xi)):
        raise ValueError('`xi` must contain finite values.')
    if xi.ndim == 1:
        if xi.shape != (2,):
            raise ValueError('`xi` with 1 dimension must have shape (2,).')
        xi = xi.reshape(1, 2)
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('`xi` with 2 dimensions must have shape (n, 2).')
    else:
        raise ValueError('`xi` must have shape (2,) or (n, 2).')
    xsi = xi[:, 0]
    eta = xi[:, 1]
    one_m_xsi = 1 - xsi
    one_p_xsi = 1 + xsi
    one_m_eta = 1 - eta
    one_p_eta = 1 + eta
    xsi_sq = xsi ** 2
    eta_sq = eta ** 2
    one_m_xsi_sq = 1 - xsi_sq
    one_m_eta_sq = 1 - eta_sq
    n_points = xi.shape[0]
    N = np.zeros((n_points, 8))
    N[:, 0] = -0.25 * one_m_xsi * one_m_eta * (1 + xsi + eta)
    N[:, 1] = 0.25 * one_p_xsi * one_m_eta * (xsi - eta - 1)
    N[:, 2] = 0.25 * one_p_xsi * one_p_eta * (xsi + eta - 1)
    N[:, 3] = 0.25 * one_m_xsi * one_p_eta * (eta - xsi - 1)
    N[:, 4] = 0.5 * one_m_xsi_sq * one_m_eta
    N[:, 5] = 0.5 * one_p_xsi * one_m_eta_sq
    N[:, 6] = 0.5 * one_m_xsi_sq * one_p_eta
    N[:, 7] = 0.5 * one_m_xsi * one_m_eta_sq
    dN_dxi = np.zeros((n_points, 8, 2))
    dN_dxi[:, 0, 0] = 0.25 * one_m_eta * (2 * xsi + eta)
    dN_dxi[:, 1, 0] = 0.25 * one_m_eta * (2 * xsi - eta)
    dN_dxi[:, 2, 0] = 0.25 * one_p_eta * (2 * xsi + eta)
    dN_dxi[:, 3, 0] = 0.25 * one_p_eta * (2 * xsi - eta)
    dN_dxi[:, 4, 0] = -xsi * one_m_eta
    dN_dxi[:, 5, 0] = 0.5 * one_m_eta_sq
    dN_dxi[:, 6, 0] = -xsi * one_p_eta
    dN_dxi[:, 7, 0] = -0.5 * one_m_eta_sq
    dN_dxi[:, 0, 1] = 0.25 * one_m_xsi * (xsi + 2 * eta)
    dN_dxi[:, 1, 1] = 0.25 * one_p_xsi * (2 * eta - xsi)
    dN_dxi[:, 2, 1] = 0.25 * one_p_xsi * (xsi + 2 * eta)
    dN_dxi[:, 3, 1] = 0.25 * one_m_xsi * (2 * eta - xsi)
    dN_dxi[:, 4, 1] = -0.5 * one_m_xsi_sq
    dN_dxi[:, 5, 1] = -eta * one_p_xsi
    dN_dxi[:, 6, 1] = 0.5 * one_m_xsi_sq
    dN_dxi[:, 7, 1] = -eta * one_m_xsi
    return (N.reshape(n_points, 8, 1), dN_dxi)
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
        raise ValueError('xi must be a NumPy array.')
    if xi.shape != (2,) and len(xi.shape) != 2 or (len(xi.shape) == 2 and xi.shape[1] != 2):
        raise ValueError('xi must have shape (2,) or (n, 2).')
    if not np.isfinite(xi).all():
        raise ValueError('xi must contain only finite values.')
    if xi.shape == (2,):
        xi = xi.reshape(1, 2)
    n = xi.shape[0]
    (xi_, eta_) = (xi[:, 0], xi[:, 1])
    N = np.zeros((n, 8, 1))
    dN_dxi = np.zeros((n, 8, 2))
    N[:, 0, 0] = -1 / 4 * (1 - xi_) * (1 - eta_) * (1 + xi_ + eta_)
    N[:, 1, 0] = 1 / 4 * (1 + xi_) * (1 - eta_) * (xi_ - eta_ - 1)
    N[:, 2, 0] = 1 / 4 * (1 + xi_) * (1 + eta_) * (xi_ + eta_ - 1)
    N[:, 3, 0] = 1 / 4 * (1 - xi_) * (1 + eta_) * (eta_ - xi_ - 1)
    N[:, 4, 0] = 1 / 2 * (1 - xi_ ** 2) * (1 - eta_)
    N[:, 5, 0] = 1 / 2 * (1 + xi_) * (1 - eta_ ** 2)
    N[:, 6, 0] = 1 / 2 * (1 - xi_ ** 2) * (1 + eta_)
    N[:, 7, 0] = 1 / 2 * (1 - xi_) * (1 - eta_ ** 2)
    dN_dxi[:, 0, 0] = -1 / 4 * (1 - eta_) * (2 * xi_ + eta_)
    dN_dxi[:, 0, 1] = -1 / 4 * (1 - xi_) * (xi_ + 2 * eta_)
    dN_dxi[:, 1, 0] = 1 / 4 * (1 - eta_) * (2 * xi_ - eta_)
    dN_dxi[:, 1, 1] = 1 / 4 * (1 + xi_) * (-xi_ + 2 * eta_)
    dN_dxi[:, 2, 0] = 1 / 4 * (1 + eta_) * (2 * xi_ + eta_)
    dN_dxi[:, 2, 1] = 1 / 4 * (1 + xi_) * (xi_ + 2 * eta_)
    dN_dxi[:, 3, 0] = 1 / 4 * (1 + eta_) * (-2 * xi_ + eta_)
    dN_dxi[:, 3, 1] = 1 / 4 * (1 - xi_) * (-xi_ + 2 * eta_)
    dN_dxi[:, 4, 0] = -xi_ * (1 - eta_)
    dN_dxi[:, 4, 1] = -1 / 2 * (1 - xi_ ** 2)
    dN_dxi[:, 5, 0] = 1 / 2 * (1 - eta_ ** 2)
    dN_dxi[:, 5, 1] = -(1 + xi_) * eta_
    dN_dxi[:, 6, 0] = -xi_ * (1 + eta_)
    dN_dxi[:, 6, 1] = 1 / 2 * (1 - xi_ ** 2)
    dN_dxi[:, 7, 0] = -1 / 2 * (1 - eta_ ** 2)
    dN_dxi[:, 7, 1] = -(1 - xi_) * eta_
    return (N, dN_dxi)
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
    if xi.ndim == 1:
        xi = xi[np.newaxis, :]
    if xi.shape[1] != 2:
        raise ValueError('`xi` must have shape (2,) or (n, 2).')
    if not np.isfinite(xi).all():
        raise ValueError('`xi` must contain only finite values.')
    xi1 = xi[:, 0]
    xi2 = xi[:, 1]
    N1 = -0.25 * (1 - xi1) * (1 - xi2) * (1 + xi1 + xi2)
    N2 = 0.25 * (1 + xi1) * (1 - xi2) * (xi1 - xi2 - 1)
    N3 = 0.25 * (1 + xi1) * (1 + xi2) * (xi1 + xi2 - 1)
    N4 = 0.25 * (1 - xi1) * (1 + xi2) * (xi2 - xi1 - 1)
    N5 = 0.5 * (1 - xi1 ** 2) * (1 - xi2)
    N6 = 0.5 * (1 + xi1) * (1 - xi2 ** 2)
    N7 = 0.5 * (1 - xi1 ** 2) * (1 + xi2)
    N8 = 0.5 * (1 - xi1) * (1 - xi2 ** 2)
    N = np.stack((N1, N2, N3, N4, N5, N6, N7, N8), axis=-1)[..., np.newaxis]
    dN1_dxi1 = 0.25 * (1 - xi2) * (2 * xi1 + xi2)
    dN1_dxi2 = 0.25 * (1 - xi1) * (xi1 + 2 * xi2)
    dN2_dxi1 = 0.25 * (1 - xi2) * (2 * xi1 - xi2)
    dN2_dxi2 = -0.25 * (1 + xi1) * (xi1 - 2 * xi2)
    dN3_dxi1 = 0.25 * (1 + xi2) * (2 * xi1 + xi2)
    dN3_dxi2 = 0.25 * (1 + xi1) * (xi1 + 2 * xi2)
    dN4_dxi1 = -0.25 * (1 + xi2) * (2 * xi1 - xi2)
    dN4_dxi2 = 0.25 * (1 - xi1) * (xi2 - 2 * xi1)
    dN5_dxi1 = -xi1 * (1 - xi2)
    dN5_dxi2 = -0.5 * (1 - xi1 ** 2)
    dN6_dxi1 = 0.5 * (1 - xi2 ** 2)
    dN6_dxi2 = -xi2 * (1 + xi1)
    dN7_dxi1 = -xi1 * (1 + xi2)
    dN7_dxi2 = 0.5 * (1 - xi1 ** 2)
    dN8_dxi1 = -0.5 * (1 - xi2 ** 2)
    dN8_dxi2 = -xi2 * (1 - xi1)
    dN_dxi = np.stack((np.stack((dN1_dxi1, dN1_dxi2), axis=-1), np.stack((dN2_dxi1, dN2_dxi2), axis=-1), np.stack((dN3_dxi1, dN3_dxi2), axis=-1), np.stack((dN4_dxi1, dN4_dxi2), axis=-1), np.stack((dN5_dxi1, dN5_dxi2), axis=-1), np.stack((dN6_dxi1, dN6_dxi2), axis=-1), np.stack((dN7_dxi1, dN7_dxi2), axis=-1), np.stack((dN8_dxi1, dN8_dxi2), axis=-1)), axis=1)
    return (N, dN_dxi)
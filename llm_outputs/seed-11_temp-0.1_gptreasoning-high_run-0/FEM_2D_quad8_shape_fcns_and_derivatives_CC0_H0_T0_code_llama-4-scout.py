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
        raise ValueError("Input 'xi' must be a NumPy array.")
    if xi.ndim not in [1, 2] or (xi.ndim == 1 and xi.shape[0] != 2) or (xi.ndim == 2 and xi.shape[1] != 2):
        raise ValueError("Input 'xi' must have shape (2,) or (n, 2).")
    if not np.all(np.isfinite(xi)):
        raise ValueError("Input 'xi' contains non-finite values.")
    n_points = xi.shape[0] if xi.ndim == 2 else 1
    xi = xi.reshape(-1, 2)
    N1 = -0.25 * (1 - xi[:, 0]) * (1 - xi[:, 1]) * (1 + xi[:, 0] + xi[:, 1])
    N2 = 0.25 * (1 + xi[:, 0]) * (1 - xi[:, 1]) * (xi[:, 0] - xi[:, 1] - 1)
    N3 = 0.25 * (1 + xi[:, 0]) * (1 + xi[:, 1]) * (xi[:, 0] + xi[:, 1] - 1)
    N4 = 0.25 * (1 - xi[:, 0]) * (1 + xi[:, 1]) * (xi[:, 1] - xi[:, 0] - 1)
    N5 = 0.5 * (1 - xi[:, 0] ** 2) * (1 - xi[:, 1])
    N6 = 0.5 * (1 + xi[:, 0]) * (1 - xi[:, 1] ** 2)
    N7 = 0.5 * (1 - xi[:, 0] ** 2) * (1 + xi[:, 1])
    N8 = 0.5 * (1 - xi[:, 0]) * (1 - xi[:, 1] ** 2)
    N = np.stack([N1, N2, N3, N4, N5, N6, N7, N8], axis=1).reshape(-1, 8, 1)
    dN1_dxi = -0.25 * ((1 - xi[:, 1]) * (1 + xi[:, 0] + xi[:, 1]) - (1 - xi[:, 0]) * (1 + xi[:, 0] + xi[:, 1]) + (1 - xi[:, 0]) * (1 - xi[:, 1]))
    dN1_deta = -0.25 * ((1 - xi[:, 0]) * (1 + xi[:, 0] + xi[:, 1]) - (1 - xi[:, 1]) * (1 + xi[:, 0] + xi[:, 1]) + (1 - xi[:, 1]) * (1 - xi[:, 0]))
    dN2_dxi = 0.25 * ((1 - xi[:, 1]) * (xi[:, 0] - xi[:, 1] - 1) + (1 + xi[:, 0]) * (1 - xi[:, 1]) + (1 + xi[:, 0]) * (xi[:, 0] - xi[:, 1] - 1))
    dN2_deta = 0.25 * (-(1 + xi[:, 0]) * (xi[:, 0] - xi[:, 1] - 1) - (1 - xi[:, 1]) * (xi[:, 0] - xi[:, 1] - 1) - (1 + xi[:, 0]) * (1 - xi[:, 1]))
    dN3_dxi = 0.25 * ((1 + xi[:, 1]) * (xi[:, 0] + xi[:, 1] - 1) + (1 + xi[:, 0]) * (1 + xi[:, 1]) + (1 + xi[:, 0]) * (xi[:, 0] + xi[:, 1] - 1))
    dN3_deta = 0.25 * ((1 + xi[:, 0]) * (xi[:, 0] + xi[:, 1] - 1) + (1 - xi[:, 0]) * (1 + xi[:, 1]) + (1 + xi[:, 0]) * (xi[:, 0] + xi[:, 1] - 1))
    dN4_dxi = 0.25 * (-(1 + xi[:, 1]) * (xi[:, 1] - xi[:, 0] - 1) - (1 - xi[:, 0]) * (1 + xi[:, 1]) + (1 - xi[:, 0]) * (xi[:, 1] - xi[:, 0] - 1))
    dN4_deta = 0.25 * ((1 - xi[:, 0]) * (xi[:, 1] - xi[:, 0] - 1) + (1 - xi[:, 1]) * (1 + xi[:, 1]) + (1 - xi[:, 0]) * (xi[:, 1] - xi[:, 0] - 1))
    dN5_dxi = -xi[:, 0] * (1 - xi[:, 1])
    dN5_deta = -0.5 * (1 - xi[:, 0] ** 2)
    dN6_dxi = 0.5 * (1 - xi[:, 1] ** 2)
    dN6_deta = -(1 + xi[:, 0]) * xi[:, 1]
    dN7_dxi = -xi[:, 0] * (1 + xi[:, 1])
    dN7_deta = 0.5 * (1 - xi[:, 0] ** 2)
    dN8_dxi = -0.5 * (1 - xi[:, 1] ** 2)
    dN8_deta = -(1 - xi[:, 0]) * xi[:, 1]
    dN_dxi = np.stack([dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi, dN5_dxi, dN6_dxi, dN7_dxi, dN8_dxi], axis=1).reshape(-1, 8, 1)
    dN_deta = np.stack([dN1_deta, dN2_deta, dN3_deta, dN4_deta, dN5_deta, dN6_deta, dN7_deta, dN8_deta], axis=1).reshape(-1, 8, 1)
    dN_dxi = np.concatenate((dN_dxi, dN_deta), axis=2)
    return (N, dN_dxi)
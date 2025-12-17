def FEM_2D_quad8_physical_gradient_CC0_H1_T3(node_coords: np.ndarray, node_values: np.ndarray, xi, eta) -> np.ndarray:
    """
    Compute the physical (x, y) gradient of a scalar field for a quadratic
    8-node quadrilateral (Q8) element at one or more natural coordinates (ξ, η).
    from nodal coordinates, and maps natural derivatives to the physical domain.
    Parameters
    ----------
    node_coords : np.ndarray
        Nodal coordinates of the Q8 element.
        Shape: (8, 2). Each row corresponds to a node, with columns [x, y].
    node_values : np.ndarray
        Scalar nodal values associated with the element.
        Shape: (8,).
    xi : float or np.ndarray
        ξ-coordinate(s) of evaluation point(s) in the reference domain [-1, 1].
        Can be a scalar or array-like of shape (n_pts,).
    eta : float or np.ndarray
        η-coordinate(s) of evaluation point(s) in the reference domain [-1, 1].
        Can be a scalar or array-like of shape (n_pts,).
    Returns
    -------
    grad_phys : np.ndarray
        Physical gradient of the scalar field at each evaluation point.
        Shape: (2, n_pts), where rows correspond to [∂u/∂x, ∂u/∂y]
        and column j corresponds to point (xi[j], eta[j]).
    Notes
    -----
        N1 = -1/4 (1-ξ)(1-η)(1+ξ+η)
        N2 =  +1/4 (1+ξ)(1-η)(ξ-η-1)
        N3 =  +1/4 (1+ξ)(1+η)(ξ+η-1)
        N4 =  +1/4 (1-ξ)(1+η)(η-ξ-1)
        N5 =  +1/2 (1-ξ²)(1-η)
        N6 =  +1/2 (1+ξ)(1-η²)
        N7 =  +1/2 (1-ξ²)(1+η)
        N8 =  +1/2 (1-ξ)(1-η²)
    Derivatives with respect to ξ and η are computed for each node in this order:
        [N1, N2, N3, N4, N5, N6, N7, N8]
    Node ordering (must match both `node_coords` and `node_values`):
        1: (-1, -1),  2: ( 1, -1),  3: ( 1,  1),  4: (-1,  1),
        5: ( 0, -1),  6: ( 1,  0),  7: ( 0,  1),  8: (-1,  0)
    """
    node_coords = np.asarray(node_coords, dtype=float)
    node_values = np.asarray(node_values, dtype=float).reshape(-1)
    if node_coords.shape != (8, 2):
        raise ValueError('node_coords must have shape (8, 2).')
    if node_values.size != 8:
        raise ValueError('node_values must contain 8 entries.')
    xi_arr = np.asarray(xi, dtype=float)
    eta_arr = np.asarray(eta, dtype=float)
    xi_b, eta_b = np.broadcast_arrays(xi_arr, eta_arr)
    s = xi_b.ravel()
    t = eta_b.ravel()
    n_pts = s.size
    one_minus_eta = 1.0 - t
    one_plus_eta = 1.0 + t
    one_minus_xi = 1.0 - s
    one_plus_xi = 1.0 + s
    xi_sq = s * s
    eta_sq = t * t
    dN1_dxi = 0.25 * one_minus_eta * (2.0 * s + t)
    dN1_deta = 0.25 * one_minus_xi * (s + 2.0 * t)
    dN2_dxi = 0.25 * one_minus_eta * (2.0 * s - t)
    dN2_deta = -0.25 * one_plus_xi * (s - 2.0 * t)
    dN3_dxi = 0.25 * one_plus_eta * (2.0 * s + t)
    dN3_deta = 0.25 * one_plus_xi * (s + 2.0 * t)
    dN4_dxi = 0.25 * one_plus_eta * (2.0 * s - t)
    dN4_deta = 0.25 * one_minus_xi * (2.0 * t - s)
    dN5_dxi = -s * one_minus_eta
    dN5_deta = -0.5 * (1.0 - xi_sq)
    dN6_dxi = 0.5 * (1.0 - eta_sq)
    dN6_deta = -t * one_plus_xi
    dN7_dxi = -s * one_plus_eta
    dN7_deta = 0.5 * (1.0 - xi_sq)
    dN8_dxi = -0.5 * (1.0 - eta_sq)
    dN8_deta = -t * one_minus_xi
    dN_dxi = np.vstack((dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi, dN5_dxi, dN6_dxi, dN7_dxi, dN8_dxi))
    dN_deta = np.vstack((dN1_deta, dN2_deta, dN3_deta, dN4_deta, dN5_deta, dN6_deta, dN7_deta, dN8_deta))
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    dx_dxi = np.dot(x, dN_dxi)
    dx_deta = np.dot(x, dN_deta)
    dy_dxi = np.dot(y, dN_dxi)
    dy_deta = np.dot(y, dN_deta)
    du_dxi = np.dot(node_values, dN_dxi)
    du_deta = np.dot(node_values, dN_deta)
    a = dx_dxi
    b = dx_deta
    c = dy_dxi
    d = dy_deta
    p = du_dxi
    q = du_deta
    det = a * d - b * c
    grad_x = (d * p - c * q) / det
    grad_y = (-b * p + a * q) / det
    grad = np.vstack((grad_x, grad_y))
    return grad
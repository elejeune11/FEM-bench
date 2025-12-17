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
    nc = np.asarray(node_coords, dtype=float)
    if nc.shape != (8, 2):
        raise ValueError('node_coords must have shape (8, 2).')
    u = np.asarray(node_values, dtype=float).reshape(-1)
    if u.size != 8:
        raise ValueError('node_values must have length 8.')
    xi_arr, eta_arr = np.broadcast_arrays(np.asarray(xi, dtype=float), np.asarray(eta, dtype=float))
    xi_flat = xi_arr.ravel()
    eta_flat = eta_arr.ravel()
    n_pts = xi_flat.size
    xi_v = xi_flat
    eta_v = eta_flat
    dN1_dxi = 0.25 * (2.0 * xi_v + eta_v) * (1.0 - eta_v)
    dN1_deta = 0.25 * (1.0 - xi_v) * (xi_v + 2.0 * eta_v)
    dN2_dxi = 0.25 * (1.0 - eta_v) * (2.0 * xi_v - eta_v)
    dN2_deta = -0.25 * (1.0 + xi_v) * (xi_v - 2.0 * eta_v)
    dN3_dxi = 0.25 * (1.0 + eta_v) * (2.0 * xi_v + eta_v)
    dN3_deta = 0.25 * (1.0 + xi_v) * (xi_v + 2.0 * eta_v)
    dN4_dxi = 0.25 * (1.0 + eta_v) * (2.0 * xi_v - eta_v)
    dN4_deta = 0.25 * (1.0 - xi_v) * (2.0 * eta_v - xi_v)
    dN5_dxi = -xi_v * (1.0 - eta_v)
    dN5_deta = -0.5 * (1.0 - xi_v * xi_v)
    dN6_dxi = 0.5 * (1.0 - eta_v * eta_v)
    dN6_deta = -(1.0 + xi_v) * eta_v
    dN7_dxi = -xi_v * (1.0 + eta_v)
    dN7_deta = 0.5 * (1.0 - xi_v * xi_v)
    dN8_dxi = -0.5 * (1.0 - eta_v * eta_v)
    dN8_deta = -(1.0 - xi_v) * eta_v
    dN_dxi = np.vstack([dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi, dN5_dxi, dN6_dxi, dN7_dxi, dN8_dxi])
    dN_deta = np.vstack([dN1_deta, dN2_deta, dN3_deta, dN4_deta, dN5_deta, dN6_deta, dN7_deta, dN8_deta])
    x = nc[:, 0]
    y = nc[:, 1]
    dx_dxi = x @ dN_dxi
    dx_deta = x @ dN_deta
    dy_dxi = y @ dN_dxi
    dy_deta = y @ dN_deta
    detJ = dx_dxi * dy_deta - dx_deta * dy_dxi
    if np.any(detJ == 0.0):
        raise np.linalg.LinAlgError('Jacobian determinant is zero at one or more evaluation points.')
    invJ11 = dy_deta / detJ
    invJ12 = -dx_deta / detJ
    invJ21 = -dy_dxi / detJ
    invJ22 = dx_dxi / detJ
    du_dxi = u @ dN_dxi
    du_deta = u @ dN_deta
    du_dx = invJ11 * du_dxi + invJ21 * du_deta
    du_dy = invJ12 * du_dxi + invJ22 * du_deta
    grad_phys = np.vstack([du_dx, du_dy])
    return grad_phys
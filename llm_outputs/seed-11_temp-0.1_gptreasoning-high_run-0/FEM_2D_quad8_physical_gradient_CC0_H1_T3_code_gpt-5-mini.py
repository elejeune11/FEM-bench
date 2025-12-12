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
    import numpy as np
    node_coords = np.asarray(node_coords, dtype=float)
    node_values = np.asarray(node_values, dtype=float)
    (xi_arr, eta_arr) = np.broadcast_arrays(np.atleast_1d(xi).astype(float), np.atleast_1d(eta).astype(float))
    xi_arr = xi_arr.ravel()
    eta_arr = eta_arr.ravel()
    n_pts = xi_arr.size
    xi_v = xi_arr
    eta_v = eta_arr
    dNdxi = np.empty((8, n_pts), dtype=float)
    dNdeta = np.empty((8, n_pts), dtype=float)
    dNdxi[0, :] = 0.25 * (1.0 - eta_v) * (2.0 * xi_v + eta_v)
    dNdeta[0, :] = 0.25 * (1.0 - xi_v) * (xi_v + 2.0 * eta_v)
    dNdxi[1, :] = 0.25 * (1.0 - eta_v) * (2.0 * xi_v - eta_v)
    dNdeta[1, :] = 0.25 * (1.0 + xi_v) * (2.0 * eta_v - xi_v)
    dNdxi[2, :] = 0.25 * (1.0 + eta_v) * (2.0 * xi_v + eta_v)
    dNdeta[2, :] = 0.25 * (1.0 + xi_v) * (xi_v + 2.0 * eta_v)
    dNdxi[3, :] = 0.25 * (1.0 + eta_v) * (2.0 * xi_v - eta_v)
    dNdeta[3, :] = 0.25 * (1.0 - xi_v) * (2.0 * eta_v - xi_v)
    dNdxi[4, :] = -xi_v * (1.0 - eta_v)
    dNdeta[4, :] = -0.5 * (1.0 - xi_v ** 2)
    dNdxi[5, :] = 0.5 * (1.0 - eta_v ** 2)
    dNdeta[5, :] = -eta_v * (1.0 + xi_v)
    dNdxi[6, :] = -xi_v * (1.0 + eta_v)
    dNdeta[6, :] = 0.5 * (1.0 - xi_v ** 2)
    dNdxi[7, :] = -0.5 * (1.0 - eta_v ** 2)
    dNdeta[7, :] = -(1.0 - xi_v) * eta_v
    x_coords = node_coords[:, 0]
    y_coords = node_coords[:, 1]
    dx_dxi = x_coords @ dNdxi
    dx_deta = x_coords @ dNdeta
    dy_dxi = y_coords @ dNdxi
    dy_deta = y_coords @ dNdeta
    du_dxi = node_values @ dNdxi
    du_deta = node_values @ dNdeta
    detJ = dx_dxi * dy_deta - dx_deta * dy_dxi
    ux = (dy_deta * du_dxi - dy_dxi * du_deta) / detJ
    uy = (-dx_deta * du_dxi + dx_dxi * du_deta) / detJ
    grad_phys = np.vstack((ux, uy))
    return grad_phys
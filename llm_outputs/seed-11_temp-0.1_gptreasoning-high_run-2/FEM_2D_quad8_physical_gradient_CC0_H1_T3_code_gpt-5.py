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
    if node_coords.shape != (8, 2):
        raise ValueError('node_coords must have shape (8, 2).')
    node_values = np.asarray(node_values, dtype=float).reshape(-1)
    if node_values.size != 8:
        raise ValueError('node_values must have shape (8,) with 8 nodal values.')
    xi_arr = np.atleast_1d(np.asarray(xi, dtype=float)).ravel()
    eta_arr = np.atleast_1d(np.asarray(eta, dtype=float)).ravel()
    if xi_arr.size == 1 and eta_arr.size > 1:
        xi_arr = np.full(eta_arr.shape, xi_arr.item(), dtype=float)
    elif eta_arr.size == 1 and xi_arr.size > 1:
        eta_arr = np.full(xi_arr.shape, eta_arr.item(), dtype=float)
    elif xi_arr.size != eta_arr.size:
        raise ValueError('xi and eta must be scalars or arrays of the same length.')
    s = xi_arr
    t = eta_arr
    n_pts = s.size
    dN1_dxi = 0.25 * (1.0 - t) * (2.0 * s + t)
    dN2_dxi = 0.25 * (1.0 - t) * (2.0 * s - t)
    dN3_dxi = 0.25 * (1.0 + t) * (2.0 * s + t)
    dN4_dxi = 0.25 * (1.0 + t) * (2.0 * s - t)
    dN5_dxi = -s * (1.0 - t)
    dN6_dxi = 0.5 * (1.0 - t ** 2)
    dN7_dxi = -s * (1.0 + t)
    dN8_dxi = -0.5 * (1.0 - t ** 2)
    dN1_deta = 0.25 * (1.0 - s) * (s + 2.0 * t)
    dN2_deta = 0.25 * (1.0 + s) * (2.0 * t - s)
    dN3_deta = 0.25 * (1.0 + s) * (s + 2.0 * t)
    dN4_deta = 0.25 * (1.0 - s) * (2.0 * t - s)
    dN5_deta = -0.5 * (1.0 - s ** 2)
    dN6_deta = -t * (1.0 + s)
    dN7_deta = 0.5 * (1.0 - s ** 2)
    dN8_deta = -t * (1.0 - s)
    dN_dxi = np.vstack([dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi, dN5_dxi, dN6_dxi, dN7_dxi, dN8_dxi])
    dN_deta = np.vstack([dN1_deta, dN2_deta, dN3_deta, dN4_deta, dN5_deta, dN6_deta, dN7_deta, dN8_deta])
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    dx_dxi = x @ dN_dxi
    dx_deta = x @ dN_deta
    dy_dxi = y @ dN_dxi
    dy_deta = y @ dN_deta
    detJ = dx_dxi * dy_deta - dx_deta * dy_dxi
    du_dxi = node_values @ dN_dxi
    du_deta = node_values @ dN_deta
    du_dx = (dy_deta * du_dxi - dx_deta * du_deta) / detJ
    du_dy = (-dy_dxi * du_dxi + dx_dxi * du_deta) / detJ
    grad_phys = np.vstack((du_dx, du_dy)).reshape(2, n_pts)
    return grad_phys
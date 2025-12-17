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
    node_values = np.asarray(node_values, dtype=float).reshape(-1)
    if node_coords.shape != (8, 2):
        raise ValueError('node_coords must have shape (8, 2).')
    if node_values.size != 8:
        raise ValueError('node_values must have shape (8,) or compatible.')
    xi_arr = np.asarray(xi, dtype=float).reshape(-1)
    eta_arr = np.asarray(eta, dtype=float).reshape(-1)
    if xi_arr.size == 1 and eta_arr.size > 1:
        xi_arr = np.full(eta_arr.shape, xi_arr.item(), dtype=float)
    elif eta_arr.size == 1 and xi_arr.size > 1:
        eta_arr = np.full(xi_arr.shape, eta_arr.item(), dtype=float)
    elif xi_arr.size != eta_arr.size:
        raise ValueError('xi and eta must have the same number of points or be scalars.')
    n_pts = xi_arr.size
    xiv = xi_arr
    etav = eta_arr
    dN1_dxi = 0.25 * (1.0 - etav) * (2.0 * xiv + etav)
    dN2_dxi = 0.25 * (1.0 - etav) * (2.0 * xiv - etav)
    dN3_dxi = 0.25 * (1.0 + etav) * (2.0 * xiv + etav)
    dN4_dxi = 0.25 * (1.0 + etav) * (2.0 * xiv - etav)
    dN5_dxi = -xiv * (1.0 - etav)
    dN6_dxi = 0.5 * (1.0 - etav ** 2)
    dN7_dxi = -xiv * (1.0 + etav)
    dN8_dxi = -0.5 * (1.0 - etav ** 2)
    dN1_deta = 0.25 * (1.0 - xiv) * (xiv + 2.0 * etav)
    dN2_deta = -0.25 * (1.0 + xiv) * (xiv - 2.0 * etav)
    dN3_deta = 0.25 * (1.0 + xiv) * (xiv + 2.0 * etav)
    dN4_deta = 0.25 * (1.0 - xiv) * (2.0 * etav - xiv)
    dN5_deta = -0.5 * (1.0 - xiv ** 2)
    dN6_deta = -(1.0 + xiv) * etav
    dN7_deta = 0.5 * (1.0 - xiv ** 2)
    dN8_deta = -(1.0 - xiv) * etav
    dN_dxi = np.column_stack([dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi, dN5_dxi, dN6_dxi, dN7_dxi, dN8_dxi])
    dN_deta = np.column_stack([dN1_deta, dN2_deta, dN3_deta, dN4_deta, dN5_deta, dN6_deta, dN7_deta, dN8_deta])
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    u = node_values
    dxdxi = dN_dxi @ x
    dxdeta = dN_deta @ x
    dydxi = dN_dxi @ y
    dydeta = dN_deta @ y
    du_dxi = dN_dxi @ u
    du_deta = dN_deta @ u
    detJ = dxdxi * dydeta - dxdeta * dydxi
    tol = 1e-14
    if np.any(np.abs(detJ) < tol):
        raise ValueError('Jacobian determinant is zero or near zero at one or more evaluation points.')
    ux = (dydeta * du_dxi - dxdeta * du_deta) / detJ
    uy = (-dydxi * du_dxi + dxdxi * du_deta) / detJ
    grad_phys = np.vstack((ux, uy))
    return grad_phys
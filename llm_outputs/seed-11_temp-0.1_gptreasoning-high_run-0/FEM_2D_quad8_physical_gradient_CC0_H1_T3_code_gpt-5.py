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
    coords = np.asarray(node_coords, dtype=float)
    vals = np.asarray(node_values, dtype=float).reshape(-1)
    if coords.shape != (8, 2):
        raise ValueError('node_coords must have shape (8, 2)')
    if vals.shape != (8,):
        raise ValueError('node_values must have shape (8,)')
    xi_arr = np.asarray(xi, dtype=float)
    eta_arr = np.asarray(eta, dtype=float)
    (xi_b, eta_b) = np.broadcast_arrays(xi_arr, eta_arr)
    r = xi_b.ravel()
    s = eta_b.ravel()
    n_pts = r.size
    one_minus_r = 1.0 - r
    one_plus_r = 1.0 + r
    one_minus_s = 1.0 - s
    one_plus_s = 1.0 + s
    r2 = r * r
    s2 = s * s
    dN1dr = 0.25 * one_minus_s * (2.0 * r + s)
    dN2dr = 0.25 * one_minus_s * (2.0 * r - s)
    dN3dr = 0.25 * one_plus_s * (2.0 * r + s)
    dN4dr = 0.25 * one_plus_s * (2.0 * r - s)
    dN5dr = -r * one_minus_s
    dN6dr = 0.5 * (1.0 - s2)
    dN7dr = -r * one_plus_s
    dN8dr = -0.5 * (1.0 - s2)
    dN1ds = 0.25 * one_minus_r * (r + 2.0 * s)
    dN2ds = -0.25 * one_plus_r * (r - 2.0 * s)
    dN3ds = 0.25 * one_plus_r * (r + 2.0 * s)
    dN4ds = 0.25 * one_minus_r * (-r + 2.0 * s)
    dN5ds = -0.5 * (1.0 - r2)
    dN6ds = -s * one_plus_r
    dN7ds = 0.5 * (1.0 - r2)
    dN8ds = -s * one_minus_r
    dNdr_mat = np.column_stack((dN1dr, dN2dr, dN3dr, dN4dr, dN5dr, dN6dr, dN7dr, dN8dr))
    dNds_mat = np.column_stack((dN1ds, dN2ds, dN3ds, dN4ds, dN5ds, dN6ds, dN7ds, dN8ds))
    x = coords[:, 0]
    y = coords[:, 1]
    dxdr = dNdr_mat @ x
    dxds = dNds_mat @ x
    dydr = dNdr_mat @ y
    dyds = dNds_mat @ y
    dudr = dNdr_mat @ vals
    duds = dNds_mat @ vals
    detJ = dxdr * dyds - dxds * dydr
    grad_x = (dyds * dudr - dydr * duds) / detJ
    grad_y = (-dxds * dudr + dxdr * duds) / detJ
    grad_phys = np.vstack((grad_x, grad_y))
    return grad_phys
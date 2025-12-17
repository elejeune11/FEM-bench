def FEM_2D_quad8_integral_of_derivative_CC0_H3_T3(node_coords: np.ndarray, node_values: np.ndarray, num_gauss_pts: int) -> np.ndarray:
    """
    Compute ∫_Ω (∇u) dΩ for a scalar field u defined over a quadratic
    8-node quadrilateral (Q8) finite element.
    The computation uses isoparametric mapping and Gauss–Legendre quadrature
    on the reference domain Q = [-1, 1] × [-1, 1].
    Parameters
    ----------
    node_coords : np.ndarray
        Physical coordinates of the Q8 element nodes.
        Shape: (8, 2). Each row is [x, y].
        Node ordering (must match both geometry and values):
            1: (-1, -1),  2: ( 1, -1),  3: ( 1,  1),  4: (-1,  1),
            5: ( 0, -1),  6: ( 1,  0),  7: ( 0,  1),  8: (-1,  0)
    node_values : np.ndarray
        Scalar nodal values of u. Shape: (8,) or (8, 1).
    num_gauss_pts : int
        Number of quadrature points to use: one of {1, 4, 9}.
    Returns
    -------
    integral : np.ndarray
        The vector [∫_Ω ∂u/∂x dΩ, ∫_Ω ∂u/∂y dΩ].
        Shape: (2,).
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
    if not isinstance(node_coords, np.ndarray) or node_coords.shape != (8, 2):
        raise ValueError('node_coords must be a numpy array with shape (8, 2).')
    if not isinstance(node_values, np.ndarray):
        raise ValueError('node_values must be a numpy array with shape (8,) or (8, 1).')
    nv = np.asarray(node_values, dtype=float).reshape(-1)
    if nv.shape[0] != 8:
        raise ValueError('node_values must contain 8 entries.')
    coords = np.asarray(node_coords, dtype=float)
    if num_gauss_pts == 1:
        gp = np.array([0.0])
        gw = np.array([2.0])
    elif num_gauss_pts == 4:
        a = 1.0 / np.sqrt(3.0)
        gp = np.array([-a, a])
        gw = np.array([1.0, 1.0])
    elif num_gauss_pts == 9:
        a = np.sqrt(3.0 / 5.0)
        gp = np.array([-a, 0.0, a])
        gw = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    else:
        raise ValueError('num_gauss_pts must be one of {1, 4, 9}.')
    integral = np.zeros(2, dtype=float)
    for i in range(len(gp)):
        xi = gp[i]
        wx = gw[i]
        for j in range(len(gp)):
            eta = gp[j]
            wy = gw[j]
            w = wx * wy
            x = xi
            e = eta
            dN_dxi = np.empty(8, dtype=float)
            dN_deta = np.empty(8, dtype=float)
            dN_dxi[0] = 0.25 * (1.0 - e) * (2.0 * x + e)
            dN_deta[0] = 0.25 * (1.0 - x) * (x + 2.0 * e)
            dN_dxi[1] = 0.25 * (1.0 - e) * (2.0 * x - e)
            dN_deta[1] = -0.25 * (1.0 + x) * (x - 2.0 * e)
            dN_dxi[2] = 0.25 * (1.0 + e) * (2.0 * x + e)
            dN_deta[2] = 0.25 * (1.0 + x) * (x + 2.0 * e)
            dN_dxi[3] = 0.25 * (1.0 + e) * (2.0 * x - e)
            dN_deta[3] = 0.25 * (1.0 - x) * (2.0 * e - x)
            dN_dxi[4] = -x * (1.0 - e)
            dN_deta[4] = -0.5 * (1.0 - x * x)
            dN_dxi[5] = 0.5 * (1.0 - e * e)
            dN_deta[5] = -(1.0 + x) * e
            dN_dxi[6] = -x * (1.0 + e)
            dN_deta[6] = 0.5 * (1.0 - x * x)
            dN_dxi[7] = -0.5 * (1.0 - e * e)
            dN_deta[7] = -(1.0 - x) * e
            dx_dxi = float(np.dot(dN_dxi, coords[:, 0]))
            dy_dxi = float(np.dot(dN_dxi, coords[:, 1]))
            dx_deta = float(np.dot(dN_deta, coords[:, 0]))
            dy_deta = float(np.dot(dN_deta, coords[:, 1]))
            detJ = dx_dxi * dy_deta - dx_deta * dy_dxi
            if np.abs(detJ) <= 1e-14:
                raise ValueError('Degenerate element: Jacobian determinant is too close to zero.')
            invJT = 1.0 / detJ * np.array([[dy_deta, -dy_dxi], [-dx_deta, dx_dxi]], dtype=float)
            du_dxi = float(np.dot(dN_dxi, nv))
            du_deta = float(np.dot(dN_deta, nv))
            grad = invJT @ np.array([du_dxi, du_deta], dtype=float)
            integral += grad * np.abs(detJ) * w
    return integral
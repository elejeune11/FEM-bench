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
    node_coords = np.asarray(node_coords, dtype=float)
    vals = np.asarray(node_values, dtype=float).reshape(-1)
    assert node_coords.shape == (8, 2)
    assert vals.size == 8
    if num_gauss_pts not in (1, 4, 9):
        raise ValueError('num_gauss_pts must be one of {1,4,9}')
    if num_gauss_pts == 1:
        gp_1d = np.array([0.0], dtype=float)
        gw_1d = np.array([2.0], dtype=float)
    elif num_gauss_pts == 4:
        a = 1.0 / np.sqrt(3.0)
        gp_1d = np.array([-a, a], dtype=float)
        gw_1d = np.array([1.0, 1.0], dtype=float)
    else:
        a = np.sqrt(3.0 / 5.0)
        gp_1d = np.array([-a, 0.0, a], dtype=float)
        gw_1d = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=float)
    integral = np.zeros(2, dtype=float)
    for (i, xi) in enumerate(gp_1d):
        for (j, eta) in enumerate(gp_1d):
            w = gw_1d[i] * gw_1d[j]
            dN_dxi = np.empty(8, dtype=float)
            dN_deta = np.empty(8, dtype=float)
            dN_dxi[0] = 0.25 * (1.0 - eta) * (2.0 * xi + eta)
            dN_deta[0] = 0.25 * (1.0 - xi) * (xi + 2.0 * eta)
            dN_dxi[1] = 0.25 * (1.0 - eta) * (2.0 * xi - eta)
            dN_deta[1] = -0.25 * (1.0 + xi) * (xi - 2.0 * eta)
            dN_dxi[2] = 0.25 * (1.0 + eta) * (2.0 * xi + eta)
            dN_deta[2] = 0.25 * (1.0 + xi) * (xi + 2.0 * eta)
            dN_dxi[3] = 0.25 * (1.0 + eta) * (2.0 * xi - eta)
            dN_deta[3] = 0.25 * (1.0 - xi) * (2.0 * eta - xi)
            dN_dxi[4] = -xi * (1.0 - eta)
            dN_deta[4] = -0.5 * (1.0 - xi * xi)
            dN_dxi[5] = 0.5 * (1.0 - eta * eta)
            dN_deta[5] = -eta * (1.0 + xi)
            dN_dxi[6] = -xi * (1.0 + eta)
            dN_deta[6] = 0.5 * (1.0 - xi * xi)
            dN_dxi[7] = -0.5 * (1.0 - eta * eta)
            dN_deta[7] = -eta * (1.0 - xi)
            J = np.zeros((2, 2), dtype=float)
            J[0, 0] = np.dot(dN_dxi, node_coords[:, 0])
            J[1, 0] = np.dot(dN_dxi, node_coords[:, 1])
            J[0, 1] = np.dot(dN_deta, node_coords[:, 0])
            J[1, 1] = np.dot(dN_deta, node_coords[:, 1])
            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)
            du_dxi = np.dot(dN_dxi, vals)
            du_deta = np.dot(dN_deta, vals)
            grad_param = np.array([du_dxi, du_deta], dtype=float)
            grad_phys = invJ.T.dot(grad_param)
            integral += grad_phys * (abs(detJ) * w)
    return integral
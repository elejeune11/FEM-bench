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
    node_values = node_values.flatten()
    if num_gauss_pts == 1:
        gauss_pts_1d = np.array([0.0])
        gauss_wts_1d = np.array([2.0])
    elif num_gauss_pts == 4:
        gauss_pts_1d = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
        gauss_wts_1d = np.array([1.0, 1.0])
    elif num_gauss_pts == 9:
        gauss_pts_1d = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        gauss_wts_1d = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    else:
        raise ValueError('num_gauss_pts must be 1, 4, or 9')

    def shape_function_derivatives(xi, eta):
        dN_dxi = np.zeros(8)
        dN_deta = np.zeros(8)
        dN_dxi[0] = -0.25 * (-(1 - eta) * (1 + xi + eta) + (1 - xi) * (1 - eta) * -1)
        dN_dxi[0] = -0.25 * ((1 - eta) * -1 * (1 + xi + eta) + (1 - xi) * (1 - eta) * 1)
        dN_dxi[0] = 0.25 * (1 - eta) * (2 * xi + eta)
        dN_deta[0] = 0.25 * (1 - xi) * (xi + 2 * eta)
        dN_dxi[1] = 0.25 * (1 - eta) * (2 * xi - eta)
        dN_deta[1] = -0.25 * (1 + xi) * (xi - 2 * eta)
        dN_dxi[2] = 0.25 * (1 + eta) * (2 * xi + eta)
        dN_deta[2] = 0.25 * (1 + xi) * (xi + 2 * eta)
        dN_dxi[3] = -0.25 * (1 + eta) * (2 * xi - eta)
        dN_deta[3] = 0.25 * (1 - xi) * (-xi + 2 * eta)
        dN_dxi[4] = -xi * (1 - eta)
        dN_deta[4] = -0.5 * (1 - xi ** 2)
        dN_dxi[5] = 0.5 * (1 - eta ** 2)
        dN_deta[5] = -(1 + xi) * eta
        dN_dxi[6] = -xi * (1 + eta)
        dN_deta[6] = 0.5 * (1 - xi ** 2)
        dN_dxi[7] = -0.5 * (1 - eta ** 2)
        dN_deta[7] = -(1 - xi) * eta
        return (dN_dxi, dN_deta)
    integral = np.zeros(2)
    for (i, xi) in enumerate(gauss_pts_1d):
        for (j, eta) in enumerate(gauss_pts_1d):
            w = gauss_wts_1d[i] * gauss_wts_1d[j]
            (dN_dxi, dN_deta) = shape_function_derivatives(xi, eta)
            dx_dxi = np.dot(dN_dxi, node_coords[:, 0])
            dx_deta = np.dot(dN_deta, node_coords[:, 0])
            dy_dxi = np.dot(dN_dxi, node_coords[:, 1])
            dy_deta = np.dot(dN_deta, node_coords[:, 1])
            J = np.array([[dx_dxi, dy_dxi], [dx_deta, dy_deta]])
            detJ = dx_dxi * dy_deta - dx_deta * dy_dxi
            invJ = np.array([[dy_deta, -dy_dxi], [-dx_deta, dx_dxi]]) / detJ
            dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
            dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
            du_dx = np.dot(dN_dx, node_values)
            du_dy = np.dot(dN_dy, node_values)
            integral[0] += du_dx * detJ * w
            integral[1] += du_dy * detJ * w
    return integral
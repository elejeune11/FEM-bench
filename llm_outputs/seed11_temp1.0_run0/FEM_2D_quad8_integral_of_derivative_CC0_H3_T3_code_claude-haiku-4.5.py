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
    node_values = np.atleast_1d(node_values).flatten()
    if num_gauss_pts == 1:
        gauss_pts = np.array([[0.0]])
        gauss_wts = np.array([2.0])
    elif num_gauss_pts == 4:
        gpt = 1.0 / np.sqrt(3.0)
        gauss_pts = np.array([[-gpt, -gpt], [gpt, -gpt], [gpt, gpt], [-gpt, gpt]])
        gauss_wts = np.array([1.0, 1.0, 1.0, 1.0])
    elif num_gauss_pts == 9:
        gpt = np.sqrt(3.0 / 5.0)
        wt1 = 5.0 / 9.0
        wt2 = 8.0 / 9.0
        pts = np.array([-gpt, 0.0, gpt])
        wts = np.array([wt1, wt2, wt1])
        gauss_pts = []
        gauss_wts = []
        for i in range(3):
            for j in range(3):
                gauss_pts.append([pts[i], pts[j]])
                gauss_wts.append(wts[i] * wts[j])
        gauss_pts = np.array(gauss_pts)
        gauss_wts = np.array(gauss_wts)
    integral = np.zeros(2)
    for gp in range(len(gauss_pts)):
        (xi, eta) = gauss_pts[gp]
        wt = gauss_wts[gp]
        N1 = -0.25 * (1 - xi) * (1 - eta) * (1 + xi + eta)
        N2 = 0.25 * (1 + xi) * (1 - eta) * (xi - eta - 1)
        N3 = 0.25 * (1 + xi) * (1 + eta) * (xi + eta - 1)
        N4 = 0.25 * (1 - xi) * (1 + eta) * (eta - xi - 1)
        N5 = 0.5 * (1 - xi * xi) * (1 - eta)
        N6 = 0.5 * (1 + xi) * (1 - eta * eta)
        N7 = 0.5 * (1 - xi * xi) * (1 + eta)
        N8 = 0.5 * (1 - xi) * (1 - eta * eta)
        N = np.array([N1, N2, N3, N4, N5, N6, N7, N8])
        dN1_dxi = -0.25 * (1 - eta) * (2 * xi + eta)
        dN2_dxi = 0.25 * (1 - eta) * (2 * xi - eta + 1)
        dN3_dxi = 0.25 * (1 + eta) * (2 * xi + eta - 1)
        dN4_dxi = 0.25 * (1 + eta) * (-2 * xi + eta + 1)
        dN5_dxi = -xi * (1 - eta)
        dN6_dxi = 0.5 * (1 - eta * eta)
        dN7_dxi = -xi * (1 + eta)
        dN8_dxi = -0.5 * (1 - eta * eta)
        dN1_deta = -0.25 * (1 - xi) * (xi + 2 * eta)
        dN2_deta = 0.25 * (1 + xi) * (-2 * eta + xi - 1)
        dN3_deta = 0.25 * (1 + xi) * (2 * eta + xi - 1)
        dN4_deta = 0.25 * (1 - xi) * (2 * eta - xi + 1)
        dN5_deta = -0.5 * (1 - xi * xi)
        dN6_deta = -(1 + xi) * eta
        dN7_deta = 0.5 * (1 - xi * xi)
        dN8_deta = -(1 - xi) * eta
        dN_dxi = np.array([dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi, dN5_dxi, dN6_dxi, dN7_dxi, dN8_dxi])
        dN_deta = np.array([dN1_deta, dN2_deta, dN3_deta, dN4_deta, dN5_deta, dN6_deta, dN7_deta, dN8_deta])
        J = np.zeros((2, 2))
        J[0, 0] = np.dot(dN_dxi, node_coords[:, 0])
        J[0, 1] = np.dot(dN_dxi, node_coords[:, 1])
        J[1, 0] = np.dot(dN_deta, node_coords[:, 0])
        J[1, 1] = np.dot(dN_deta, node_coords[:, 1])
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        J_inv = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / detJ
        dN_dx = J_inv[0, 0] * dN_dxi + J_inv[0, 1] * dN_deta
        dN_dy = J_inv[1, 0] * dN_dxi + J_inv[1, 1] * dN_deta
        du_dx = np.dot(dN_dx, node_values)
        du_dy = np.dot(dN_dy, node_values)
        integral[0] += du_dx * np.abs(detJ) * wt
        integral[1] += du_dy * np.abs(detJ) * wt
    return integral
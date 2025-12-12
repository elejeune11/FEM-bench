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
        gauss_pts = np.array([[0.0]])
        gauss_wts = np.array([2.0])
    elif num_gauss_pts == 4:
        pt = 1.0 / np.sqrt(3.0)
        gauss_pts = np.array([[-pt, -pt], [pt, -pt], [pt, pt], [-pt, pt]])
        gauss_wts = np.array([1.0, 1.0, 1.0, 1.0])
    elif num_gauss_pts == 9:
        pt = np.sqrt(3.0 / 5.0)
        gauss_pts = np.array([[-pt, -pt], [0.0, -pt], [pt, -pt], [-pt, 0.0], [0.0, 0.0], [pt, 0.0], [-pt, pt], [0.0, pt], [pt, pt]])
        gauss_wts = np.array([25.0 / 81.0, 40.0 / 81.0, 25.0 / 81.0, 40.0 / 81.0, 64.0 / 81.0, 40.0 / 81.0, 25.0 / 81.0, 40.0 / 81.0, 25.0 / 81.0])
    integral = np.zeros(2)
    for gp in range(len(gauss_pts)):
        (xi, eta) = gauss_pts[gp]
        w = gauss_wts[gp]
        N = np.array([-0.25 * (1 - xi) * (1 - eta) * (1 + xi + eta), 0.25 * (1 + xi) * (1 - eta) * (xi - eta - 1), 0.25 * (1 + xi) * (1 + eta) * (xi + eta - 1), 0.25 * (1 - xi) * (1 + eta) * (eta - xi - 1), 0.5 * (1 - xi ** 2) * (1 - eta), 0.5 * (1 + xi) * (1 - eta ** 2), 0.5 * (1 - xi ** 2) * (1 + eta), 0.5 * (1 - xi) * (1 - eta ** 2)])
        dN_dxi = np.array([-0.25 * (-(1 - eta) * (1 + xi + eta) + (1 - xi) * (1 - eta)), 0.25 * ((1 - eta) * (xi - eta - 1) + (1 + xi) * (1 - eta)), 0.25 * ((1 + eta) * (xi + eta - 1) + (1 + xi) * (1 + eta)), 0.25 * (-(1 + eta) * (eta - xi - 1) + (1 - xi) * (1 + eta)), 0.5 * (-2 * xi) * (1 - eta), 0.5 * (1 - eta ** 2), 0.5 * (-2 * xi) * (1 + eta), 0.5 * -(1 - eta ** 2)])
        dN_deta = np.array([-0.25 * ((1 - xi) * (1 + xi + eta) + (1 - xi) * (1 - eta)), 0.25 * ((1 + xi) * (-(xi - eta - 1) - (1 - eta))), 0.25 * ((1 + xi) * (1 + eta) + (1 + xi) * (xi + eta - 1)), 0.25 * ((1 - xi) * (1 + eta) + (1 - xi) * (eta - xi - 1)), 0.5 * (1 - xi ** 2) * -1, 0.5 * (1 + xi) * (-2 * eta), 0.5 * (1 - xi ** 2), 0.5 * (1 - xi) * (-2 * eta)])
        J = np.zeros((2, 2))
        J[0, 0] = np.dot(dN_dxi, node_coords[:, 0])
        J[0, 1] = np.dot(dN_dxi, node_coords[:, 1])
        J[1, 0] = np.dot(dN_deta, node_coords[:, 0])
        J[1, 1] = np.dot(dN_deta, node_coords[:, 1])
        det_J = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        J_inv = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / det_J
        dN_dx = J_inv[0, 0] * dN_dxi + J_inv[0, 1] * dN_deta
        dN_dy = J_inv[1, 0] * dN_dxi + J_inv[1, 1] * dN_deta
        du_dx = np.dot(dN_dx, node_values)
        du_dy = np.dot(dN_dy, node_values)
        integral[0] += w * du_dx * det_J
        integral[1] += w * du_dy * det_J
    return integral
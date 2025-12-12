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
    if num_gauss_pts == 1:
        gauss_pts = np.array([[0.0, 0.0]])
        gauss_wts = np.array([4.0])
    elif num_gauss_pts == 4:
        gauss_pts = np.array([[-1 / np.sqrt(3), -1 / np.sqrt(3)], [1 / np.sqrt(3), -1 / np.sqrt(3)], [1 / np.sqrt(3), 1 / np.sqrt(3)], [-1 / np.sqrt(3), 1 / np.sqrt(3)]])
        gauss_wts = np.array([1.0, 1.0, 1.0, 1.0])
    elif num_gauss_pts == 9:
        gauss_pts = np.array([[-np.sqrt(3 / 5), -np.sqrt(3 / 5)], [0.0, -np.sqrt(3 / 5)], [np.sqrt(3 / 5), -np.sqrt(3 / 5)], [-np.sqrt(3 / 5), 0.0], [0.0, 0.0], [np.sqrt(3 / 5), 0.0], [-np.sqrt(3 / 5), np.sqrt(3 / 5)], [0.0, np.sqrt(3 / 5)], [np.sqrt(3 / 5), np.sqrt(3 / 5)]])
        gauss_wts = np.array([25 / 81, 40 / 81, 25 / 81, 40 / 81, 64 / 81, 40 / 81, 25 / 81, 40 / 81, 25 / 81])
    else:
        raise ValueError('num_gauss_pts must be one of {1, 4, 9}')
    integral = np.zeros(2)
    for i in range(gauss_pts.shape[0]):
        (xi, eta) = gauss_pts[i, :]
        N = np.array([-1 / 4 * (1 - xi) * (1 - eta) * (1 + xi + eta), 1 / 4 * (1 + xi) * (1 - eta) * (xi - eta - 1), 1 / 4 * (1 + xi) * (1 + eta) * (xi + eta - 1), 1 / 4 * (1 - xi) * (1 + eta) * (eta - xi - 1), 1 / 2 * (1 - xi ** 2) * (1 - eta), 1 / 2 * (1 + xi) * (1 - eta ** 2), 1 / 2 * (1 - xi ** 2) * (1 + eta), 1 / 2 * (1 - xi) * (1 - eta ** 2)])
        dN_dxi = np.array([[-1 / 4 * (-1 + eta) * (1 + 2 * xi + eta)], [1 / 4 * (1 - eta) * (2 * xi - eta - 1)], [1 / 4 * (1 + eta) * (2 * xi + eta - 1)], [1 / 4 * (-1 - eta) * (-2 * xi + eta - 1)], [-xi * (1 - eta)], [1 / 2 * (1 - eta ** 2)], [-xi * (1 + eta)], [-1 / 2 * (1 - eta ** 2)]])
        dN_deta = np.array([[-1 / 4 * (-1 + xi) * (1 + xi + 2 * eta)], [1 / 4 * (-1 - xi) * (xi - 2 * eta - 1)], [1 / 4 * (1 + xi) * (xi + 2 * eta - 1)], [1 / 4 * (1 - xi) * (-xi + 2 * eta - 1)], [-1 / 2 * (1 - xi ** 2)], [-(1 + xi) * eta], [1 / 2 * (1 - xi ** 2)], [-(1 - xi) * eta]])
        J = np.column_stack((np.dot(node_coords[:, 0], dN_dxi), np.dot(node_coords[:, 0], dN_deta), np.dot(node_coords[:, 1], dN_dxi), np.dot(node_coords[:, 1], dN_deta))).reshape(2, 2)
        det_J = np.linalg.det(J)
        inv_J = np.linalg.inv(J)
        dN_dx = np.dot(inv_J[0, 0], dN_dxi) + np.dot(inv_J[0, 1], dN_deta)
        dN_dy = np.dot(inv_J[1, 0], dN_dxi) + np.dot(inv_J[1, 1], dN_deta)
        integral[0] += gauss_wts[i] * np.dot(node_values.flatten(), dN_dx) * det_J
        integral[1] += gauss_wts[i] * np.dot(node_values.flatten(), dN_dy) * det_J
    return integral
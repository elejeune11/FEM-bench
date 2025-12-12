def FEM_2D_quad8_integral_of_derivative_CC0_H3_T3(node_coords: np.ndarray, node_values: np.ndarray, num_gauss_pts: int) -> np.ndarray:
    if num_gauss_pts == 1:
        gauss_pts = np.array([[0.0, 0.0]])
        weights = np.array([4.0])
    elif num_gauss_pts == 4:
        sqrt3 = np.sqrt(3)
        gauss_pts = np.array([[-1 / sqrt3, -1 / sqrt3], [1 / sqrt3, -1 / sqrt3], [1 / sqrt3, 1 / sqrt3], [-1 / sqrt3, 1 / sqrt3]])
        weights = np.array([1.0, 1.0, 1.0, 1.0])
    elif num_gauss_pts == 9:
        sqrt3_5 = np.sqrt(3 / 5)
        gauss_pts = np.array([[-sqrt3_5, -sqrt3_5], [0.0, -sqrt3_5], [sqrt3_5, -sqrt3_5], [-sqrt3_5, 0.0], [0.0, 0.0], [sqrt3_5, 0.0], [-sqrt3_5, sqrt3_5], [0.0, sqrt3_5], [sqrt3_5, sqrt3_5]])
        weights = np.array([25 / 81, 40 / 81, 25 / 81, 40 / 81, 64 / 81, 40 / 81, 25 / 81, 40 / 81, 25 / 81])
    else:
        raise ValueError('num_gauss_pts must be 1, 4, or 9')
    integral = np.zeros(2)
    for i in range(len(gauss_pts)):
        (xi, eta) = gauss_pts[i]
        w = weights[i]
        dN1_dxi = -1 / 4 * -1 * (1 - eta) * (1 + xi + eta) + -1 / 4 * (1 - xi) * (1 - eta) * 1
        dN1_deta = -1 / 4 * (1 - xi) * -1 * (1 + xi + eta) + -1 / 4 * (1 - xi) * (1 - eta) * 1
        dN2_dxi = 1 / 4 * 1 * (1 - eta) * (xi - eta - 1) + 1 / 4 * (1 + xi) * (1 - eta) * 1
        dN2_deta = 1 / 4 * (1 + xi) * -1 * (xi - eta - 1) + 1 / 4 * (1 + xi) * (1 - eta) * -1
        dN3_dxi = 1 / 4 * 1 * (1 + eta) * (xi + eta - 1) + 1 / 4 * (1 + xi) * (1 + eta) * 1
        dN3_deta = 1 / 4 * (1 + xi) * 1 * (xi + eta - 1) + 1 / 4 * (1 + xi) * (1 + eta) * 1
        dN4_dxi = 1 / 4 * -1 * (1 + eta) * (eta - xi - 1) + 1 / 4 * (1 - xi) * (1 + eta) * -1
        dN4_deta = 1 / 4 * (1 - xi) * 1 * (eta - xi - 1) + 1 / 4 * (1 - xi) * (1 + eta) * 1
        dN5_dxi = 1 / 2 * (-2 * xi) * (1 - eta)
        dN5_deta = 1 / 2 * (1 - xi ** 2) * -1
        dN6_dxi = 1 / 2 * 1 * (1 - eta ** 2)
        dN6_deta = 1 / 2 * (1 + xi) * (-2 * eta)
        dN7_dxi = 1 / 2 * (-2 * xi) * (1 + eta)
        dN7_deta = 1 / 2 * (1 - xi ** 2) * 1
        dN8_dxi = 1 / 2 * -1 * (1 - eta ** 2)
        dN8_deta = 1 / 2 * (1 - xi) * (-2 * eta)
        dN_dxi = np.array([dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi, dN5_dxi, dN6_dxi, dN7_dxi, dN8_dxi])
        dN_deta = np.array([dN1_deta, dN2_deta, dN3_deta, dN4_deta, dN5_deta, dN6_deta, dN7_deta, dN8_deta])
        J = np.zeros((2, 2))
        J[0, 0] = np.dot(dN_dxi, node_coords[:, 0])
        J[0, 1] = np.dot(dN_dxi, node_coords[:, 1])
        J[1, 0] = np.dot(dN_deta, node_coords[:, 0])
        J[1, 1] = np.dot(dN_deta, node_coords[:, 1])
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(detJ) < 1e-12:
            invJ = np.zeros((2, 2))
        else:
            invJ = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / detJ
        dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
        dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
        du_dx = np.dot(dN_dx, node_values)
        du_dy = np.dot(dN_dy, node_values)
        integral[0] += du_dx * detJ * w
        integral[1] += du_dy * detJ * w
    return integral
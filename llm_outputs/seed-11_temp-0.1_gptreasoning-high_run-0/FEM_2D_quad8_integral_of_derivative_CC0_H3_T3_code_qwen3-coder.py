def FEM_2D_quad8_integral_of_derivative_CC0_H3_T3(node_coords: np.ndarray, node_values: np.ndarray, num_gauss_pts: int) -> np.ndarray:
    if num_gauss_pts == 1:
        gauss_points = np.array([[0.0, 0.0]])
        gauss_weights = np.array([4.0])
    elif num_gauss_pts == 4:
        gp_1d = np.array([-np.sqrt(1 / 3), np.sqrt(1 / 3)])
        gauss_points = np.array([[xi, eta] for xi in gp_1d for eta in gp_1d])
        gauss_weights = np.array([1.0] * 4)
    elif num_gauss_pts == 9:
        gp_1d = np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)])
        gauss_points = np.array([[xi, eta] for xi in gp_1d for eta in gp_1d])
        gauss_weights = np.array([w1 * w2 for w1 in [5 / 9, 8 / 9, 5 / 9] for w2 in [5 / 9, 8 / 9, 5 / 9]])
    else:
        raise ValueError('num_gauss_pts must be 1, 4, or 9')

    def shape_derivatives(xi, eta):
        dN_dxi = np.array([0.25 * (2 * xi + eta) * (1 - eta), 0.25 * (2 * xi - eta) * (1 - eta), 0.25 * (2 * xi + eta) * (1 + eta), 0.25 * (2 * xi - eta) * (1 + eta), -xi * (1 - eta), 0.5 * (1 - eta ** 2), -xi * (1 + eta), -0.5 * (1 - eta ** 2)])
        dN_deta = np.array([0.25 * (xi + 2 * eta) * (1 - xi), 0.25 * (xi - 2 * eta) * (1 + xi), 0.25 * (xi + 2 * eta) * (1 + xi), 0.25 * (xi - 2 * eta) * (1 - xi), -0.5 * (1 - xi ** 2), -eta * (1 + xi), 0.5 * (1 - xi ** 2), -eta * (1 - xi)])
        return (dN_dxi, dN_deta)
    integral = np.zeros(2)
    node_values_flat = node_values.flatten()
    for i in range(len(gauss_points)):
        (xi, eta) = gauss_points[i]
        weight = gauss_weights[i]
        (dN_dxi, dN_deta) = shape_derivatives(xi, eta)
        J = np.array([[np.dot(dN_dxi, node_coords[:, 0]), np.dot(dN_deta, node_coords[:, 0])], [np.dot(dN_dxi, node_coords[:, 1]), np.dot(dN_deta, node_coords[:, 1])]])
        detJ = np.linalg.det(J)
        if detJ <= 0:
            raise ValueError('Non-positive Jacobian determinant')
        J_inv = np.linalg.inv(J)
        du_dxi = np.dot(dN_dxi, node_values_flat)
        du_deta = np.dot(dN_deta, node_values_flat)
        grad_u = np.dot(J_inv, np.array([du_dxi, du_deta]))
        integral += weight * detJ * grad_u
    return integral
def compute_integral_of_derivative_quad8(node_coords: np.ndarray, node_values: np.ndarray, num_gauss_pts: int) -> np.ndarray:
    node_values = np.squeeze(node_values)
    (points, weights) = quad_quadrature_2D(num_gauss_pts)
    xi_points = points[:, 0]
    eta_points = points[:, 1]
    grad_phys = compute_physical_gradient_quad8(node_coords, node_values, xi_points, eta_points)
    (_, dN) = quad8_shape_functions_and_derivatives(points)
    integral = np.zeros(2)
    for i in range(len(weights)):
        dN_i = dN[i]
        J = node_coords.T @ dN_i
        detJ = np.linalg.det(J)
        integral += grad_phys[:, i] * weights[i] * detJ
    return integral
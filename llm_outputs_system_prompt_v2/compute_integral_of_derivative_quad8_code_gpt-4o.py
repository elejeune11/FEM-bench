def compute_integral_of_derivative_quad8(node_coords: np.ndarray, node_values: np.ndarray, num_gauss_pts: int) -> np.ndarray:
    (points, weights) = quad_quadrature_2D(num_gauss_pts)
    n_pts = points.shape[0]
    integral = np.zeros(2, dtype=float)
    for i in range(n_pts):
        (xi, eta) = points[i]
        grad_phys = compute_physical_gradient_quad8(node_coords, node_values, xi, eta)
        (_, dN) = quad8_shape_functions_and_derivatives(np.array([[xi, eta]]))
        J = node_coords.T @ dN[0]
        detJ = np.linalg.det(J)
        integral += weights[i] * grad_phys[:, 0] * detJ
    return integral
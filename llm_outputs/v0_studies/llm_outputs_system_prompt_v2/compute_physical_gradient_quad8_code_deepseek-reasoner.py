def compute_physical_gradient_quad8(node_coords: np.ndarray, node_values: np.ndarray, xi, eta) -> np.ndarray:
    xi_arr = np.atleast_1d(xi)
    eta_arr = np.atleast_1d(eta)
    points = np.column_stack([xi_arr.ravel(), eta_arr.ravel()])
    (_, dN_dxi) = quad8_shape_functions_and_derivatives(points)
    n_pts = points.shape[0]
    grad_phys = np.empty((2, n_pts))
    for i in range(n_pts):
        J = dN_dxi[i].T @ node_coords
        grad_nat = dN_dxi[i].T @ node_values
        grad_phys[:, i] = np.linalg.solve(J.T, grad_nat)
    return grad_phys.reshape(2, *xi_arr.shape)
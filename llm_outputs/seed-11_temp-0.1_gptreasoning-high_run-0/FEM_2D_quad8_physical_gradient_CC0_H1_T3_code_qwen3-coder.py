def FEM_2D_quad8_physical_gradient_CC0_H1_T3(node_coords: np.ndarray, node_values: np.ndarray, xi, eta) -> np.ndarray:
    xi = np.atleast_1d(xi)
    eta = np.atleast_1d(eta)
    n_pts = len(xi)
    grad_phys = np.zeros((2, n_pts))
    dN_dxi = np.array([1 / 4 * (1 - eta) * (2 * xi + eta), 1 / 4 * (1 - eta) * (2 * xi - eta), 1 / 4 * (1 + eta) * (2 * xi + eta), 1 / 4 * (1 + eta) * (2 * xi - eta), -xi * (1 - eta), 1 / 2 * (1 - eta ** 2), -xi * (1 + eta), -1 / 2 * (1 - eta ** 2)])
    dN_deta = np.array([1 / 4 * (1 - xi) * (xi + 2 * eta), -1 / 4 * (1 + xi) * (xi - 2 * eta), 1 / 4 * (1 + xi) * (xi + 2 * eta), -1 / 4 * (1 - xi) * (xi - 2 * eta), -1 / 2 * (1 - xi ** 2), -eta * (1 + xi), 1 / 2 * (1 - xi ** 2), -eta * (1 - xi)])
    for i in range(n_pts):
        dN_dxi_pt = dN_dxi[:, i]
        dN_deta_pt = dN_deta[:, i]
        J = np.zeros((2, 2))
        for j in range(8):
            J[0, 0] += dN_dxi_pt[j] * node_coords[j, 0]
            J[0, 1] += dN_deta_pt[j] * node_coords[j, 0]
            J[1, 0] += dN_dxi_pt[j] * node_coords[j, 1]
            J[1, 1] += dN_deta_pt[j] * node_coords[j, 1]
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        du_dxi = np.dot(dN_dxi_pt, node_values)
        du_deta = np.dot(dN_deta_pt, node_values)
        dxi_dx = J[1, 1] / detJ
        deta_dx = -J[0, 1] / detJ
        dxi_dy = -J[1, 0] / detJ
        deta_dy = J[0, 0] / detJ
        du_dx = dxi_dx * du_dxi + deta_dx * du_deta
        du_dy = dxi_dy * du_dxi + deta_dy * du_deta
        grad_phys[:, i] = [du_dx, du_dy]
    return grad_phys
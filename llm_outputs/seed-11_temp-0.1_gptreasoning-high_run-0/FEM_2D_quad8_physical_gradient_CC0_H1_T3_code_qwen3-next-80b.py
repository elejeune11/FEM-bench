def FEM_2D_quad8_physical_gradient_CC0_H1_T3(node_coords: np.ndarray, node_values: np.ndarray, xi, eta) -> np.ndarray:
    xi = np.asarray(xi)
    eta = np.asarray(eta)
    if xi.ndim == 0:
        xi = np.array([xi])
        eta = np.array([eta])
    n_pts = len(xi)
    dN1_dxi = -1 / 4 * (-1 * (1 - eta) * (1 + xi + eta) + (1 - xi) * (1 - eta) * 1)
    dN1_deta = -1 / 4 * ((1 - xi) * -1 * (1 + xi + eta) + (1 - xi) * (1 - eta) * 1)
    dN2_dxi = 1 / 4 * (1 * (1 - eta) * (xi - eta - 1) + (1 + xi) * (1 - eta) * 1)
    dN2_deta = 1 / 4 * ((1 + xi) * -1 * (xi - eta - 1) + (1 + xi) * (1 - eta) * -1)
    dN3_dxi = 1 / 4 * (1 * (1 + eta) * (xi + eta - 1) + (1 + xi) * (1 + eta) * 1)
    dN3_deta = 1 / 4 * ((1 + xi) * 1 * (xi + eta - 1) + (1 + xi) * (1 + eta) * 1)
    dN4_dxi = 1 / 4 * (-1 * (1 + eta) * (eta - xi - 1) + (1 - xi) * (1 + eta) * -1)
    dN4_deta = 1 / 4 * ((1 - xi) * 1 * (eta - xi - 1) + (1 - xi) * (1 + eta) * 1)
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
    grad_ref = np.zeros((2, n_pts))
    for i in range(n_pts):
        dN_dxi_i = dN_dxi[:, i] if dN_dxi.ndim > 1 else dN_dxi
        dN_deta_i = dN_deta[:, i] if dN_deta.ndim > 1 else dN_deta
        dx_dxi = np.dot(dN_dxi_i, node_coords[:, 0])
        dx_deta = np.dot(dN_deta_i, node_coords[:, 0])
        dy_dxi = np.dot(dN_dxi_i, node_coords[:, 1])
        dy_deta = np.dot(dN_deta_i, node_coords[:, 1])
        J = np.array([[dx_dxi, dx_deta], [dy_dxi, dy_deta]])
        det_J = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(det_J) < 1e-15:
            J_inv = np.zeros_like(J)
        else:
            J_inv = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / det_J
        dN_dxi_i = dN_dxi_i.reshape(-1, 1)
        dN_deta_i = dN_deta_i.reshape(-1, 1)
        grad_ref_i = np.array([np.dot(dN_dxi_i.T, node_values), np.dot(dN_deta_i.T, node_values)])
        grad_phys_i = J_inv @ grad_ref_i
        grad_ref[:, i] = grad_phys_i.flatten()
    return grad_ref
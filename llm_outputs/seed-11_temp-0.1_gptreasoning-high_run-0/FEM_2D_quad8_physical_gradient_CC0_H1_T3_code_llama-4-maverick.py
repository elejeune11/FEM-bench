def FEM_2D_quad8_physical_gradient_CC0_H1_T3(node_coords: np.ndarray, node_values: np.ndarray, xi, eta) -> np.ndarray:
    """
    Compute the physical (x, y) gradient of a scalar field for a quadratic
    8-node quadrilateral (Q8) element at one or more natural coordinates (ξ, η).
    from nodal coordinates, and maps natural derivatives to the physical domain.
    Parameters
    ----------
    node_coords : np.ndarray
        Nodal coordinates of the Q8 element.
        Shape: (8, 2). Each row corresponds to a node, with columns [x, y].
    node_values : np.ndarray
        Scalar nodal values associated with the element.
        Shape: (8,).
    xi : float or np.ndarray
        ξ-coordinate(s) of evaluation point(s) in the reference domain [-1, 1].
        Can be a scalar or array-like of shape (n_pts,).
    eta : float or np.ndarray
        η-coordinate(s) of evaluation point(s) in the reference domain [-1, 1].
        Can be a scalar or array-like of shape (n_pts,).
    Returns
    -------
    grad_phys : np.ndarray
        Physical gradient of the scalar field at each evaluation point.
        Shape: (2, n_pts), where rows correspond to [∂u/∂x, ∂u/∂y]
        and column j corresponds to point (xi[j], eta[j]).
    Notes
    -----
        N1 = -1/4 (1-ξ)(1-η)(1+ξ+η)
        N2 =  +1/4 (1+ξ)(1-η)(ξ-η-1)
        N3 =  +1/4 (1+ξ)(1+η)(ξ+η-1)
        N4 =  +1/4 (1-ξ)(1+η)(η-ξ-1)
        N5 =  +1/2 (1-ξ²)(1-η)
        N6 =  +1/2 (1+ξ)(1-η²)
        N7 =  +1/2 (1-ξ²)(1+η)
        N8 =  +1/2 (1-ξ)(1-η²)
    Derivatives with respect to ξ and η are computed for each node in this order:
        [N1, N2, N3, N4, N5, N6, N7, N8]
    Node ordering (must match both `node_coords` and `node_values`):
        1: (-1, -1),  2: ( 1, -1),  3: ( 1,  1),  4: (-1,  1),
        5: ( 0, -1),  6: ( 1,  0),  7: ( 0,  1),  8: (-1,  0)
    """
    xi = np.atleast_1d(xi)
    eta = np.atleast_1d(eta)
    n_pts = max(len(xi), len(eta))
    if len(xi) == 1:
        xi = np.repeat(xi, n_pts)
    if len(eta) == 1:
        eta = np.repeat(eta, n_pts)
    dN_dxi = np.array([-0.25 * (1 - eta) * (1 + 2 * xi + eta), 0.25 * (1 - eta) * (2 * xi - eta - 1), 0.25 * (1 + eta) * (2 * xi + eta - 1), -0.25 * (1 + eta) * (-2 * xi + eta + 1), -xi * (1 - eta), 0.5 * (1 - eta ** 2), -xi * (1 + eta), -0.5 * (1 - eta ** 2)])
    dN_deta = np.array([-0.25 * (1 - xi) * (1 + xi + 2 * eta), -0.25 * (1 + xi) * (-1 + xi - 2 * eta), 0.25 * (1 + xi) * (2 * eta + xi + 1), 0.25 * (1 - xi) * (2 * eta - xi + 1), -0.5 * (1 - xi ** 2), -eta * (1 + xi), 0.5 * (1 - xi ** 2), -eta * (1 - xi)])
    dN_dxi = dN_dxi[:, :, np.newaxis] * np.ones((1, 1, n_pts))
    dN_deta = dN_deta[:, :, np.newaxis] * np.ones((1, 1, n_pts))
    for i in range(8):
        dN_dxi[i, :, :] = dN_dxi[i, :, :] * np.ones(n_pts)
        dN_deta[i, :, :] = dN_deta[i, :, :] * np.ones(n_pts)
    dN_dxi_eval = np.sum(dN_dxi * xi[np.newaxis, np.newaxis, :] ** np.arange(8)[:, np.newaxis, np.newaxis], axis=0)
    dN_deta_eval = np.sum(dN_deta * eta[np.newaxis, np.newaxis, :] ** np.arange(8)[:, np.newaxis, np.newaxis], axis=0)
    J = np.zeros((2, 2, n_pts))
    J[0, 0, :] = np.sum(node_coords[:, 0][:, np.newaxis] * dN_dxi_eval, axis=0)
    J[0, 1, :] = np.sum(node_coords[:, 0][:, np.newaxis] * dN_deta_eval, axis=0)
    J[1, 0, :] = np.sum(node_coords[:, 1][:, np.newaxis] * dN_dxi_eval, axis=0)
    J[1, 1, :] = np.sum(node_coords[:, 1][:, np.newaxis] * dN_deta_eval, axis=0)
    det_J = J[0, 0, :] * J[1, 1, :] - J[0, 1, :] * J[1, 0, :]
    J_inv = np.zeros_like(J)
    J_inv[0, 0, :] = J[1, 1, :] / det_J
    J_inv[0, 1, :] = -J[0, 1, :] / det_J
    J_inv[1, 0, :] = -J[1, 0, :] / det_J
    J_inv[1, 1, :] = J[0, 0, :] / det_J
    grad_nat = np.zeros((2, n_pts))
    grad_nat[0, :] = np.sum(node_values[:, np.newaxis] * dN_dxi_eval, axis=0)
    grad_nat[1, :] = np.sum(node_values[:, np.newaxis] * dN_deta_eval, axis=0)
    grad_phys = np.zeros((2, n_pts))
    grad_phys[0, :] = J_inv[0, 0, :] * grad_nat[0, :] + J_inv[0, 1, :] * grad_nat[1, :]
    grad_phys[1, :] = J_inv[1, 0, :] * grad_nat[0, :] + J_inv[1, 1, :] * grad_nat[1, :]
    return grad_phys
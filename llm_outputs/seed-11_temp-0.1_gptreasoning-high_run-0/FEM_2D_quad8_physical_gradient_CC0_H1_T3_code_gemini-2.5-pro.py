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
    n_pts = xi.shape[0]
    xi2 = xi * xi
    eta2 = eta * eta
    dN_dxi = np.zeros((8, n_pts))
    dN_dxi[0, :] = 0.25 * (1.0 - eta) * (2.0 * xi + eta)
    dN_dxi[1, :] = 0.25 * (1.0 - eta) * (2.0 * xi - eta)
    dN_dxi[2, :] = 0.25 * (1.0 + eta) * (2.0 * xi + eta)
    dN_dxi[3, :] = 0.25 * (1.0 + eta) * (2.0 * xi - eta)
    dN_dxi[4, :] = -xi * (1.0 - eta)
    dN_dxi[5, :] = 0.5 * (1.0 - eta2)
    dN_dxi[6, :] = -xi * (1.0 + eta)
    dN_dxi[7, :] = -0.5 * (1.0 - eta2)
    dN_deta = np.zeros((8, n_pts))
    dN_deta[0, :] = 0.25 * (1.0 - xi) * (xi + 2.0 * eta)
    dN_deta[1, :] = 0.25 * (1.0 + xi) * (-xi + 2.0 * eta)
    dN_deta[2, :] = 0.25 * (1.0 + xi) * (xi + 2.0 * eta)
    dN_deta[3, :] = 0.25 * (1.0 - xi) * (-xi + 2.0 * eta)
    dN_deta[4, :] = -0.5 * (1.0 - xi2)
    dN_deta[5, :] = -eta * (1.0 + xi)
    dN_deta[6, :] = 0.5 * (1.0 - xi2)
    dN_deta[7, :] = -eta * (1.0 - xi)
    dN_dxi_eta = np.stack((dN_dxi, dN_deta), axis=0)
    J = np.einsum('ijk,jl->ilk', dN_dxi_eta, node_coords)
    det_J = J[0, 0, :] * J[1, 1, :] - J[0, 1, :] * J[1, 0, :]
    inv_det_J = 1.0 / det_J
    J_inv = np.empty_like(J)
    J_inv[0, 0, :] = J[1, 1, :] * inv_det_J
    J_inv[0, 1, :] = -J[0, 1, :] * inv_det_J
    J_inv[1, 0, :] = -J[1, 0, :] * inv_det_J
    J_inv[1, 1, :] = J[0, 0, :] * inv_det_J
    grad_nat = np.einsum('ijk,j->ik', dN_dxi_eta, node_values)
    grad_phys = np.einsum('ijk,jk->ik', J_inv, grad_nat)
    return grad_phys
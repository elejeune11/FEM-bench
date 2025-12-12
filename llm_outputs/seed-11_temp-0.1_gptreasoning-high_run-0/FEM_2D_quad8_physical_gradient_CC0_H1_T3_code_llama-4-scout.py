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
    xi = np.asarray(xi)
    eta = np.asarray(eta)
    n_pts = xi.size
    dNxi = (-0.25 * (1.0 - eta) * (1.0 + eta + xi), 0.25 * (1.0 - eta) * (1.0 + xi - eta), 0.25 * (1.0 + eta) * (1.0 + xi + eta), -0.25 * (1.0 + eta) * (1.0 - xi + eta), -xi * (1.0 - eta), 0.5 * (1.0 - eta ** 2), -xi * (1.0 + eta), -0.5 * (1.0 - eta ** 2))
    deta = (-0.25 * (1.0 - xi) * (1.0 + xi + eta), -0.25 * (1.0 + xi) * (xi - eta - 1.0), 0.25 * (1.0 + xi) * (xi + eta - 1.0), 0.25 * (1.0 - xi) * (eta - xi - 1.0), -0.5 * (1.0 - xi ** 2), -eta * (1.0 + xi), 0.5 * (1.0 - xi ** 2), -eta * (1.0 - xi))
    dNxi = np.array([f(xi, eta) for f in dNxi])
    deta = np.array([f(xi, eta) for f in deta])
    dNxi = dNxi.T
    deta = deta.T
    J = np.vstack((np.dot(node_coords[:, 0], dNxi), np.dot(node_coords[:, 1], deta)))
    J_inv = np.linalg.inv(J)
    grad_nat = np.dot(np.array([dNxi, deta]), node_values)
    grad_phys = np.dot(J_inv, grad_nat)
    return grad_phys
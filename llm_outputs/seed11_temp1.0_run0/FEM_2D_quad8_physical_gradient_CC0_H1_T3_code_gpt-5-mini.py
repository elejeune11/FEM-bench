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
    """
    xi_arr = np.atleast_1d(xi).astype(float)
    eta_arr = np.atleast_1d(eta).astype(float)
    if xi_arr.shape != eta_arr.shape:
        xi_arr = np.broadcast_to(xi_arr, eta_arr.shape)
    n_pts = xi_arr.size
    grad_phys = np.zeros((2, n_pts), dtype=float)
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    u = node_values.reshape(8)
    for i in range(n_pts):
        a = xi_arr.flat[i]
        b = eta_arr.flat[i]
        dN1_dxi = 0.25 * (1.0 - b) * (2.0 * a + b)
        dN2_dxi = 0.25 * (1.0 - b) * (2.0 * a - b)
        dN3_dxi = 0.25 * (1.0 + b) * (2.0 * a + b)
        dN4_dxi = 0.25 * (1.0 + b) * (2.0 * a - b)
        dN5_dxi = -a * (1.0 - b)
        dN6_dxi = 0.5 * (1.0 - b * b)
        dN7_dxi = -a * (1.0 + b)
        dN8_dxi = -0.5 * (1.0 - b * b)
        dN_dxi = np.array([dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi, dN5_dxi, dN6_dxi, dN7_dxi, dN8_dxi], dtype=float)
        dN1_deta = 0.25 * (1.0 - a) * (a + 2.0 * b)
        dN2_deta = 0.25 * (1.0 + a) * (-a + 2.0 * b)
        dN3_deta = 0.25 * (1.0 + a) * (a + 2.0 * b)
        dN4_deta = 0.25 * (1.0 - a) * (-a + 2.0 * b)
        dN5_deta = -0.5 * (1.0 - a * a)
        dN6_deta = -(1.0 + a) * b
        dN7_deta = 0.5 * (1.0 - a * a)
        dN8_deta = -(1.0 - a) * b
        dN_deta = np.array([dN1_deta, dN2_deta, dN3_deta, dN4_deta, dN5_deta, dN6_deta, dN7_deta, dN8_deta], dtype=float)
        du_dxi = np.dot(dN_dxi, u)
        du_deta = np.dot(dN_deta, u)
        grad_nat = np.array([du_dxi, du_deta], dtype=float)
        dx_dxi = np.dot(dN_dxi, x)
        dx_deta = np.dot(dN_deta, x)
        dy_dxi = np.dot(dN_dxi, y)
        dy_deta = np.dot(dN_deta, y)
        J = np.array([[dx_dxi, dx_deta], [dy_dxi, dy_deta]], dtype=float)
        invJ = np.linalg.inv(J)
        grad_phys[:, i] = invJ.T.dot(grad_nat)
    return grad_phys
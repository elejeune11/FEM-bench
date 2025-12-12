def FEM_2D_quad8_integral_of_derivative_CC0_H3_T3(node_coords: np.ndarray, node_values: np.ndarray, num_gauss_pts: int) -> np.ndarray:
    """
    Compute ∫_Ω (∇u) dΩ for a scalar field u defined over a quadratic
    8-node quadrilateral (Q8) finite element.
    The computation uses isoparametric mapping and Gauss–Legendre quadrature
    on the reference domain Q = [-1, 1] × [-1, 1].
    Parameters
    ----------
    node_coords : np.ndarray
        Physical coordinates of the Q8 element nodes.
        Shape: (8, 2). Each row is [x, y].
        Node ordering (must match both geometry and values):
            1: (-1, -1),  2: ( 1, -1),  3: ( 1,  1),  4: (-1,  1),
            5: ( 0, -1),  6: ( 1,  0),  7: ( 0,  1),  8: (-1,  0)
    node_values : np.ndarray
        Scalar nodal values of u. Shape: (8,) or (8, 1).
    num_gauss_pts : int
        Number of quadrature points to use: one of {1, 4, 9}.
    Returns
    -------
    integral : np.ndarray
        The vector [∫_Ω ∂u/∂x dΩ, ∫_Ω ∂u/∂y dΩ].
        Shape: (2,).
    Notes
    -----
    Shape functions:
        N1 = -1/4 (1-ξ)(1-η)(1+ξ+η)
        N2 =  +1/4 (1+ξ)(1-η)(ξ-η-1)
        N3 =  +1/4 (1+ξ)(1+η)(ξ+η-1)
        N4 =  +1/4 (1-ξ)(1+η)(η-ξ-1)
        N5 =  +1/2 (1-ξ²)(1-η)
        N6 =  +1/2 (1+ξ)(1-η²)
        N7 =  +1/2 (1-ξ²)(1+η)
        N8 =  +1/2 (1-ξ)(1-η²)
    """
    if node_coords is None or node_values is None:
        pytest.fail('node_coords and node_values must be provided')
    node_coords = np.asarray(node_coords)
    node_values = np.asarray(node_values).ravel()
    if node_coords.shape != (8, 2):
        pytest.fail('node_coords must have shape (8,2)')
    if node_values.size != 8:
        pytest.fail('node_values must have length 8')
    if num_gauss_pts not in (1, 4, 9):
        pytest.fail('num_gauss_pts must be one of {1,4,9}')
    if num_gauss_pts == 1:
        xi_1d = np.array([0.0])
        w_1d = np.array([2.0])
    elif num_gauss_pts == 4:
        a = 1.0 / np.sqrt(3.0)
        xi_1d = np.array([-a, a])
        w_1d = np.array([1.0, 1.0])
    else:
        a = np.sqrt(3.0 / 5.0)
        xi_1d = np.array([-a, 0.0, a])
        w_1d = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    xi_list = []
    eta_list = []
    weights = []
    for (i, xi) in enumerate(xi_1d):
        for (j, eta) in enumerate(xi_1d):
            xi_list.append(xi)
            eta_list.append(eta)
            weights.append(w_1d[i] * w_1d[j])
    xi_list = np.array(xi_list)
    eta_list = np.array(eta_list)
    weights = np.array(weights)
    integral = np.zeros(2, dtype=float)
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    u_vals = node_values.flatten()
    for (xi, eta, w) in zip(xi_list, eta_list, weights):
        dN = np.zeros((8, 2), dtype=float)
        dN[0, 0] = (1.0 - eta) * (2.0 * xi + eta) / 4.0
        dN[0, 1] = (1.0 - xi) * (xi + 2.0 * eta) / 4.0
        dN[1, 0] = (1.0 - eta) * (2.0 * xi - eta) / 4.0
        dN[1, 1] = -(1.0 + xi) * (xi - 2.0 * eta) / 4.0
        dN[2, 0] = (1.0 + eta) * (2.0 * xi + eta) / 4.0
        dN[2, 1] = (1.0 + xi) * (xi + 2.0 * eta) / 4.0
        dN[3, 0] = -(1.0 + eta) * (eta - 2.0 * xi) / 4.0
        dN[3, 1] = (1.0 - xi) * (2.0 * eta - xi) / 4.0
        dN[4, 0] = -xi * (1.0 - eta)
        dN[4, 1] = -0.5 * (1.0 - xi * xi)
        dN[5, 0] = 0.5 * (1.0 - eta * eta)
        dN[5, 1] = -(1.0 + xi) * eta
        dN[6, 0] = -xi * (1.0 + eta)
        dN[6, 1] = 0.5 * (1.0 - xi * xi)
        dN[7, 0] = -0.5 * (1.0 - eta * eta)
        dN[7, 1] = -(1.0 - xi) * eta
        dx_dxi = np.dot(dN[:, 0], x)
        dx_deta = np.dot(dN[:, 1], x)
        dy_dxi = np.dot(dN[:, 0], y)
        dy_deta = np.dot(dN[:, 1], y)
        J = np.array([[dx_dxi, dx_deta], [dy_dxi, dy_deta]], dtype=float)
        detJ = np.linalg.det(J)
        if detJ == 0.0:
            pytest.fail('Jacobian determinant is zero at a Gauss point')
        du_dxi = np.dot(dN[:, 0], u_vals)
        du_deta = np.dot(dN[:, 1], u_vals)
        vec_xi = np.array([du_dxi, du_deta], dtype=float)
        J_inv = np.linalg.inv(J)
        grad_phys = J_inv.T.dot(vec_xi)
        integral += grad_phys * detJ * w
    return integral
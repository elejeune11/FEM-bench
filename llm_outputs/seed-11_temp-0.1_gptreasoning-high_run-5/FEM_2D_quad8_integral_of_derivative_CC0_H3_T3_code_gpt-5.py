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
    import numpy as np
    coords = np.asarray(node_coords, dtype=float)
    if coords.shape != (8, 2):
        raise ValueError('node_coords must have shape (8, 2).')
    values = np.asarray(node_values, dtype=float).reshape(-1)
    if values.shape[0] != 8:
        raise ValueError('node_values must have shape (8,) or (8, 1).')
    if num_gauss_pts not in (1, 4, 9):
        raise ValueError('num_gauss_pts must be one of {1, 4, 9}.')
    n1d = int(round(np.sqrt(num_gauss_pts)))
    if n1d == 1:
        gp = np.array([0.0], dtype=float)
        gw = np.array([2.0], dtype=float)
    elif n1d == 2:
        a = 1.0 / np.sqrt(3.0)
        gp = np.array([-a, a], dtype=float)
        gw = np.array([1.0, 1.0], dtype=float)
    else:
        a = np.sqrt(3.0 / 5.0)
        gp = np.array([-a, 0.0, a], dtype=float)
        gw = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=float)
    x = coords[:, 0]
    y = coords[:, 1]
    integral = np.zeros(2, dtype=float)
    for i, xi in enumerate(gp):
        wi = gw[i]
        for j, eta in enumerate(gp):
            wj = gw[j]
            dN_dxi = np.empty(8, dtype=float)
            dN_deta = np.empty(8, dtype=float)
            dN_dxi[0] = 0.25 * (1.0 - eta) * (2.0 * xi + eta)
            dN_deta[0] = 0.25 * (1.0 - xi) * (xi + 2.0 * eta)
            dN_dxi[1] = 0.25 * (1.0 - eta) * (2.0 * xi - eta)
            dN_deta[1] = 0.25 * (1.0 + xi) * (2.0 * eta - xi)
            dN_dxi[2] = 0.25 * (1.0 + eta) * (2.0 * xi + eta)
            dN_deta[2] = 0.25 * (1.0 + xi) * (xi + 2.0 * eta)
            dN_dxi[3] = 0.25 * (1.0 + eta) * (2.0 * xi - eta)
            dN_deta[3] = 0.25 * (1.0 - xi) * (2.0 * eta - xi)
            dN_dxi[4] = -xi * (1.0 - eta)
            dN_deta[4] = -0.5 * (1.0 - xi * xi)
            dN_dxi[5] = 0.5 * (1.0 - eta * eta)
            dN_deta[5] = -(1.0 + xi) * eta
            dN_dxi[6] = -xi * (1.0 + eta)
            dN_deta[6] = 0.5 * (1.0 - xi * xi)
            dN_dxi[7] = -0.5 * (1.0 - eta * eta)
            dN_deta[7] = -(1.0 - xi) * eta
            J11 = float(np.dot(dN_dxi, x))
            J12 = float(np.dot(dN_deta, x))
            J21 = float(np.dot(dN_dxi, y))
            J22 = float(np.dot(dN_deta, y))
            J = np.array([[J11, J12], [J21, J22]], dtype=float)
            du_dxi = float(np.dot(dN_dxi, values))
            du_deta = float(np.dot(dN_deta, values))
            grad_ref = np.array([du_dxi, du_deta], dtype=float)
            try:
                grad_global = np.linalg.solve(J.T, grad_ref)
            except np.linalg.LinAlgError:
                raise ValueError('Jacobian is singular at a quadrature point.')
            detJ = J11 * J22 - J12 * J21
            weight = wi * wj * abs(detJ)
            integral += weight * grad_global
    return integral
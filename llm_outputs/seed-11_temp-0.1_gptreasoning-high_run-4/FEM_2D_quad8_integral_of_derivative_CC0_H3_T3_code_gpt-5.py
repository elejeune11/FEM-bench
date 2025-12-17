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
    coords = np.asarray(node_coords, dtype=float)
    if coords.shape != (8, 2):
        raise ValueError('node_coords must have shape (8, 2).')
    vals = np.asarray(node_values, dtype=float)
    if vals.ndim == 2 and vals.shape == (8, 1):
        vals = vals.reshape(8)
    if vals.shape != (8,):
        raise ValueError('node_values must have shape (8,) or (8, 1).')
    if num_gauss_pts not in (1, 4, 9):
        raise ValueError('num_gauss_pts must be one of {1, 4, 9}.')
    if num_gauss_pts == 1:
        xi_pts = np.array([0.0])
        xi_wts = np.array([2.0])
    elif num_gauss_pts == 4:
        a = 1.0 / np.sqrt(3.0)
        xi_pts = np.array([-a, a])
        xi_wts = np.array([1.0, 1.0])
    else:
        a = np.sqrt(3.0 / 5.0)
        xi_pts = np.array([-a, 0.0, a])
        xi_wts = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    integral = np.zeros(2, dtype=float)
    for i in range(xi_pts.size):
        xi = xi_pts[i]
        wxi = xi_wts[i]
        for j in range(xi_pts.size):
            eta = xi_pts[j]
            weta = xi_wts[j]
            dN_dxi = np.empty(8, dtype=float)
            dN_deta = np.empty(8, dtype=float)
            dN_dxi[0] = 0.25 * (1.0 - eta) * (2.0 * xi + eta)
            dN_deta[0] = 0.25 * (1.0 - xi) * (xi + 2.0 * eta)
            dN_dxi[1] = 0.25 * (1.0 - eta) * (2.0 * xi - eta)
            dN_deta[1] = -0.25 * (1.0 + xi) * (xi - 2.0 * eta)
            dN_dxi[2] = 0.25 * (1.0 + eta) * (2.0 * xi + eta)
            dN_deta[2] = 0.25 * (1.0 + xi) * (xi + 2.0 * eta)
            dN_dxi[3] = 0.25 * (1.0 + eta) * (2.0 * xi - eta)
            dN_deta[3] = 0.25 * (1.0 - xi) * (-xi + 2.0 * eta)
            dN_dxi[4] = -xi * (1.0 - eta)
            dN_deta[4] = -0.5 * (1.0 - xi * xi)
            dN_dxi[5] = 0.5 * (1.0 - eta * eta)
            dN_deta[5] = -(1.0 + xi) * eta
            dN_dxi[6] = -xi * (1.0 + eta)
            dN_deta[6] = 0.5 * (1.0 - xi * xi)
            dN_dxi[7] = -0.5 * (1.0 - eta * eta)
            dN_deta[7] = -(1.0 - xi) * eta
            J11 = float(np.dot(dN_dxi, coords[:, 0]))
            J12 = float(np.dot(dN_deta, coords[:, 0]))
            J21 = float(np.dot(dN_dxi, coords[:, 1]))
            J22 = float(np.dot(dN_deta, coords[:, 1]))
            detJ = J11 * J22 - J12 * J21
            if not np.isfinite(detJ) or abs(detJ) < 1e-14:
                raise ValueError('Jacobian determinant is near zero or invalid at a quadrature point.')
            inv_detJ = 1.0 / detJ
            invJ00 = J22 * inv_detJ
            invJ01 = -J12 * inv_detJ
            invJ10 = -J21 * inv_detJ
            invJ11 = J11 * inv_detJ
            du_dxi = float(np.dot(dN_dxi, vals))
            du_deta = float(np.dot(dN_deta, vals))
            grad_x = invJ00 * du_dxi + invJ10 * du_deta
            grad_y = invJ01 * du_dxi + invJ11 * du_deta
            w = wxi * weta
            integral[0] += grad_x * detJ * w
            integral[1] += grad_y * detJ * w
    return integral
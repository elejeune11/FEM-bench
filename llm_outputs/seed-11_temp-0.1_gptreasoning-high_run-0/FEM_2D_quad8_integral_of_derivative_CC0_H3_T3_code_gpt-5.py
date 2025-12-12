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
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.shape != (8, 2):
        raise ValueError('node_coords must have shape (8, 2)')
    u = np.asarray(node_values, dtype=float)
    if u.shape == (8, 1):
        u = u.ravel()
    if u.shape != (8,):
        raise ValueError('node_values must have shape (8,) or (8, 1)')
    if num_gauss_pts not in (1, 4, 9):
        raise ValueError('num_gauss_pts must be one of {1, 4, 9}')
    order_map = {1: 1, 4: 2, 9: 3}
    n = order_map[num_gauss_pts]
    if n == 1:
        gp = np.array([0.0])
        gw = np.array([2.0])
    elif n == 2:
        a = 1.0 / np.sqrt(3.0)
        gp = np.array([-a, a])
        gw = np.array([1.0, 1.0])
    else:
        a = np.sqrt(3.0 / 5.0)
        gp = np.array([-a, 0.0, a])
        gw = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    integral = np.zeros(2, dtype=float)

    def shape_derivatives(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
        dN_dxi = np.empty(8, dtype=float)
        dN_deta = np.empty(8, dtype=float)
        dN_dxi[0] = 0.25 * (1.0 - eta) * (2.0 * xi + eta)
        dN_dxi[1] = 0.25 * (1.0 - eta) * (2.0 * xi - eta)
        dN_dxi[2] = 0.25 * (1.0 + eta) * (2.0 * xi + eta)
        dN_dxi[3] = 0.25 * (1.0 + eta) * (2.0 * xi - eta)
        dN_deta[0] = 0.25 * (1.0 - xi) * (xi + 2.0 * eta)
        dN_deta[1] = -0.25 * (1.0 + xi) * (xi - 2.0 * eta)
        dN_deta[2] = 0.25 * (1.0 + xi) * (xi + 2.0 * eta)
        dN_deta[3] = 0.25 * (1.0 - xi) * (2.0 * eta - xi)
        dN_dxi[4] = -xi * (1.0 - eta)
        dN_dxi[5] = 0.5 * (1.0 - eta * eta)
        dN_dxi[6] = -xi * (1.0 + eta)
        dN_dxi[7] = -0.5 * (1.0 - eta * eta)
        dN_deta[4] = -0.5 * (1.0 - xi * xi)
        dN_deta[5] = -(1.0 + xi) * eta
        dN_deta[6] = 0.5 * (1.0 - xi * xi)
        dN_deta[7] = -(1.0 - xi) * eta
        return (dN_dxi, dN_deta)
    for i in range(n):
        xi = gp[i]
        wi = gw[i]
        for j in range(n):
            eta = gp[j]
            wj = gw[j]
            w = wi * wj
            (dN_dxi, dN_deta) = shape_derivatives(xi, eta)
            dx_dxi = float(np.dot(dN_dxi, x))
            dx_deta = float(np.dot(dN_deta, x))
            dy_dxi = float(np.dot(dN_dxi, y))
            dy_deta = float(np.dot(dN_deta, y))
            detJ = dx_dxi * dy_deta - dx_deta * dy_dxi
            if not np.isfinite(detJ) or abs(detJ) < 1e-14:
                raise ValueError('Jacobian determinant is non-finite or near zero.')
            invJ = 1.0 / detJ * np.array([[dy_deta, -dx_deta], [-dy_dxi, dx_dxi]], dtype=float)
            dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
            dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
            du_dx = float(np.dot(u, dN_dx))
            du_dy = float(np.dot(u, dN_dy))
            integral[0] += du_dx * detJ * w
            integral[1] += du_dy * detJ * w
    return integral
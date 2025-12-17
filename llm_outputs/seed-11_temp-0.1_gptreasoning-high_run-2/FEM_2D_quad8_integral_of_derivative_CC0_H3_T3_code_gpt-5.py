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
    nc = np.asarray(node_coords, dtype=float)
    if nc.shape != (8, 2):
        raise ValueError('node_coords must have shape (8, 2)')
    uv = np.asarray(node_values, dtype=float).reshape(-1)
    if uv.shape[0] != 8:
        raise ValueError('node_values must have length 8')
    if num_gauss_pts == 1:
        pts = np.array([0.0])
        wts = np.array([2.0])
    elif num_gauss_pts == 4:
        a = 1.0 / np.sqrt(3.0)
        pts = np.array([-a, a])
        wts = np.array([1.0, 1.0])
    elif num_gauss_pts == 9:
        a = np.sqrt(3.0 / 5.0)
        pts = np.array([-a, 0.0, a])
        wts = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    else:
        raise ValueError('num_gauss_pts must be one of {1, 4, 9}')
    x = nc[:, 0]
    y = nc[:, 1]
    integral = np.zeros(2, dtype=float)
    tol = 1e-12
    for i, xi in enumerate(pts):
        wx = wts[i]
        for j, eta in enumerate(pts):
            wy = wts[j]
            w2 = wx * wy
            dN1_dxi = 0.25 * (1.0 - eta) * (2.0 * xi + eta)
            dN1_deta = 0.25 * (1.0 - xi) * (xi + 2.0 * eta)
            dN2_dxi = 0.25 * (1.0 - eta) * (2.0 * xi - eta)
            dN2_deta = -0.25 * (1.0 + xi) * (xi - 2.0 * eta)
            dN3_dxi = 0.25 * (1.0 + eta) * (2.0 * xi + eta)
            dN3_deta = 0.25 * (1.0 + xi) * (xi + 2.0 * eta)
            dN4_dxi = 0.25 * (1.0 + eta) * (2.0 * xi - eta)
            dN4_deta = 0.25 * (1.0 - xi) * (2.0 * eta - xi)
            dN5_dxi = -xi * (1.0 - eta)
            dN5_deta = -0.5 * (1.0 - xi * xi)
            dN6_dxi = 0.5 * (1.0 - eta * eta)
            dN6_deta = -(1.0 + xi) * eta
            dN7_dxi = -xi * (1.0 + eta)
            dN7_deta = 0.5 * (1.0 - xi * xi)
            dN8_dxi = -0.5 * (1.0 - eta * eta)
            dN8_deta = -(1.0 - xi) * eta
            dN_dxi = np.array([dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi, dN5_dxi, dN6_dxi, dN7_dxi, dN8_dxi])
            dN_deta = np.array([dN1_deta, dN2_deta, dN3_deta, dN4_deta, dN5_deta, dN6_deta, dN7_deta, dN8_deta])
            x_xi = float(np.dot(dN_dxi, x))
            x_eta = float(np.dot(dN_deta, x))
            y_xi = float(np.dot(dN_dxi, y))
            y_eta = float(np.dot(dN_deta, y))
            detJ = x_xi * y_eta - x_eta * y_xi
            detJ_abs = abs(detJ)
            if detJ_abs <= tol:
                raise ValueError('Jacobian determinant is near zero at a quadrature point.')
            u_xi = float(np.dot(dN_dxi, uv))
            u_eta = float(np.dot(dN_deta, uv))
            gradx = (y_eta * u_xi - y_xi * u_eta) / detJ
            grady = (-x_eta * u_xi + x_xi * u_eta) / detJ
            integral[0] += gradx * detJ_abs * w2
            integral[1] += grady * detJ_abs * w2
    return integral
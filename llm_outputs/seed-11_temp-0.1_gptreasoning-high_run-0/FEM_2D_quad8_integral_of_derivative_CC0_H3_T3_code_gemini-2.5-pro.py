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
    if num_gauss_pts == 1:
        gauss_pts = np.array([0.0])
        gauss_wts = np.array([2.0])
    elif num_gauss_pts == 4:
        p = 1.0 / np.sqrt(3.0)
        gauss_pts = np.array([-p, p])
        gauss_wts = np.array([1.0, 1.0])
    elif num_gauss_pts == 9:
        p = np.sqrt(3.0 / 5.0)
        gauss_pts = np.array([-p, 0.0, p])
        gauss_wts = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    else:
        raise ValueError('num_gauss_pts must be one of {1, 4, 9}')

    def get_shape_derivatives(xi, eta):
        """Computes the derivatives of the 8-node shape functions."""
        dN_dxi = np.array([0.25 * (1 - eta) * (2 * xi + eta), 0.25 * (1 - eta) * (2 * xi - eta), 0.25 * (1 + eta) * (2 * xi + eta), 0.25 * (1 + eta) * (2 * xi - eta), -xi * (1 - eta), 0.5 * (1 - eta ** 2), -xi * (1 + eta), -0.5 * (1 - eta ** 2)])
        dN_deta = np.array([0.25 * (1 - xi) * (xi + 2 * eta), 0.25 * (1 + xi) * (-xi + 2 * eta), 0.25 * (1 + xi) * (xi + 2 * eta), 0.25 * (1 - xi) * (-xi + 2 * eta), -0.5 * (1 - xi ** 2), -eta * (1 + xi), 0.5 * (1 - xi ** 2), -eta * (1 - xi)])
        return np.vstack((dN_dxi, dN_deta)).T
    integral = np.zeros(2)
    node_vals_flat = node_values.flatten()
    for i in range(len(gauss_pts)):
        for j in range(len(gauss_pts)):
            xi_gp = gauss_pts[i]
            eta_gp = gauss_pts[j]
            weight = gauss_wts[i] * gauss_wts[j]
            B_hat = get_shape_derivatives(xi_gp, eta_gp)
            J = B_hat.T @ node_coords
            detJ = np.linalg.det(J)
            grad_u_ref = B_hat.T @ node_vals_flat
            grad_u_phys = np.linalg.solve(J, grad_u_ref)
            integral += grad_u_phys * detJ * weight
    return integral
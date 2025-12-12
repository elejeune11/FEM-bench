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
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.shape != (8, 2):
        raise ValueError('node_coords must have shape (8, 2).')
    vals = np.asarray(node_values, dtype=float).reshape(-1)
    if vals.shape[0] != 8:
        raise ValueError('node_values must have 8 entries (shape (8,) or (8,1)).')
    if num_gauss_pts not in (1, 4, 9):
        raise ValueError('num_gauss_pts must be one of {1, 4, 9}.')
    if num_gauss_pts == 1:
        gp = np.array([0.0])
        gw = np.array([2.0])
    elif num_gauss_pts == 4:
        a = 1.0 / np.sqrt(3.0)
        gp = np.array([-a, a])
        gw = np.array([1.0, 1.0])
    else:
        a = np.sqrt(3.0 / 5.0)
        gp = np.array([-a, 0.0, a])
        gw = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

    def dN_dr_ds(r: float, s: float) -> Tuple[np.ndarray, np.ndarray]:
        one_minus_s = 1.0 - s
        one_plus_s = 1.0 + s
        one_minus_r = 1.0 - r
        one_plus_r = 1.0 + r
        one_minus_s2 = 1.0 - s * s
        one_minus_r2 = 1.0 - r * r
        dNdr = np.empty(8, dtype=float)
        dNdr[0] = 0.25 * one_minus_s * (2.0 * r + s)
        dNdr[1] = 0.25 * one_minus_s * (2.0 * r - s)
        dNdr[2] = 0.25 * one_plus_s * (2.0 * r + s)
        dNdr[3] = 0.25 * one_plus_s * (2.0 * r - s)
        dNdr[4] = -r * one_minus_s
        dNdr[5] = 0.5 * one_minus_s2
        dNdr[6] = -r * one_plus_s
        dNdr[7] = -0.5 * one_minus_s2
        dNds = np.empty(8, dtype=float)
        dNds[0] = 0.25 * one_minus_r * (r + 2.0 * s)
        dNds[1] = 0.25 * one_plus_r * (2.0 * s - r)
        dNds[2] = 0.25 * one_plus_r * (r + 2.0 * s)
        dNds[3] = 0.25 * one_minus_r * (2.0 * s - r)
        dNds[4] = -0.5 * one_minus_r2
        dNds[5] = -(1.0 + r) * s
        dNds[6] = 0.5 * one_minus_r2
        dNds[7] = -(1.0 - r) * s
        return (dNdr, dNds)
    integral = np.zeros(2, dtype=float)
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    for (i, (ri, wi)) in enumerate(zip(gp, gw)):
        for (j, (sj, wj)) in enumerate(zip(gp, gw)):
            (dNdr, dNds) = dN_dr_ds(ri, sj)
            j11 = np.dot(dNdr, x)
            j12 = np.dot(dNds, x)
            j21 = np.dot(dNdr, y)
            j22 = np.dot(dNds, y)
            J = np.array([[j11, j12], [j21, j22]], dtype=float)
            detJ = np.linalg.det(J)
            if detJ == 0.0:
                raise ValueError('Singular Jacobian encountered at a quadrature point.')
            invJ = np.linalg.inv(J)
            dN_param = np.vstack((dNdr, dNds))
            dN_phys = invJ.T @ dN_param
            grad_u = dN_phys @ vals
            w = wi * wj
            integral += grad_u * (abs(detJ) * w)
    return integral
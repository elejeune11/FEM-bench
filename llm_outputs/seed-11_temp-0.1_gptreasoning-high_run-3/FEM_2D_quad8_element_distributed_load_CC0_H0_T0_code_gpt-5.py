def FEM_2D_quad8_element_distributed_load_CC0_H0_T0(face: int, node_coords: np.ndarray, traction: np.ndarray, num_gauss_pts: int) -> np.ndarray:
    """
    Assemble the consistent nodal load vector for a single edge of a Q8
    (8-node quadratic quadrilateral) under a constant traction and
    return it scattered into the full element DOF vector of length 16:
    [Fx1, Fy1, Fx2, Fy2, …, Fx8, Fy8].
    Traction model
    --------------
    `traction` is a constant Cauchy traction (force per unit physical
    edge length) applied along the current edge:
        traction = [t_x, t_y]  (shape (2,))
    Expected Q8 node ordering (must match `node_coords`)
    ----------------------------------------------------
        1:(-1,-1), 2:( 1,-1), 3:( 1, 1), 4:(-1, 1),
        5:( 0,-1), 6:( 1, 0), 7:( 0, 1), 8:(-1, 0)
    Face orientation & edge connectivity (start, mid, end)
    ------------------------------------------------------
        face=0 (bottom): (0, 4, 1)
        face=1 (right) : (1, 5, 2)
        face=2 (top)   : (2, 6, 3)
        face=3 (left)  : (3, 7, 0)
    The local edge parameter s runs from the start corner (s=-1) to the
    end corner (s=+1), with the mid-edge node at s=0.
    Parameters
    ----------
    face : {0,1,2,3}
        Which edge to load (bottom, right, top, left in the reference element).
    node_coords : (8,2) float array
        Physical coordinates of the Q8 nodes in the expected ordering above.
    traction : (2,) float array
        Constant Cauchy traction vector [t_x, t_y].
    num_gauss_pts : {1,2,3}
        1D Gauss–Legendre points on [-1,1]. (For straight edges,
        2-pt is exact with constant traction.)
    Returns
    -------
    r_elem : (16,) float array
        Element load vector in DOF order [Fx1, Fy1, Fx2, Fy2, …, Fx8, Fy8].
        Only the three edge nodes receive nonzero entries; others are zero.
    """
    if face not in (0, 1, 2, 3):
        raise ValueError('face must be one of {0,1,2,3}')
    if num_gauss_pts not in (1, 2, 3):
        raise ValueError('num_gauss_pts must be one of {1,2,3}')
    X = np.asarray(node_coords, dtype=float)
    if X.shape != (8, 2):
        raise ValueError('node_coords must have shape (8,2)')
    t = np.asarray(traction, dtype=float).reshape(2)
    r_elem = np.zeros(16, dtype=float)
    edge_nodes = [(0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0)]
    if num_gauss_pts == 1:
        gp = np.array([0.0])
        gw = np.array([2.0])
    elif num_gauss_pts == 2:
        r3 = 1.0 / np.sqrt(3.0)
        gp = np.array([-r3, r3])
        gw = np.array([1.0, 1.0])
    else:
        r5 = np.sqrt(3.0 / 5.0)
        gp = np.array([-r5, 0.0, r5])
        gw = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

    def shape_Q8(xi: float, eta: float):
        A1 = 1.0 - xi
        B1 = 1.0 - eta
        C1 = -xi - eta - 1.0
        N1 = 0.25 * A1 * B1 * C1
        dN1_dxi = -0.25 * B1 * (C1 + A1)
        dN1_deta = -0.25 * A1 * (C1 + B1)
        A2 = 1.0 + xi
        B2 = 1.0 - eta
        C2 = xi - eta - 1.0
        N2 = 0.25 * A2 * B2 * C2
        dN2_dxi = 0.25 * B2 * (C2 + A2)
        dN2_deta = -0.25 * A2 * (C2 + B2)
        A3 = 1.0 + xi
        B3 = 1.0 + eta
        C3 = xi + eta - 1.0
        N3 = 0.25 * A3 * B3 * C3
        dN3_dxi = 0.25 * B3 * (C3 + A3)
        dN3_deta = 0.25 * A3 * (C3 + B3)
        A4 = 1.0 - xi
        B4 = 1.0 + eta
        C4 = -xi + eta - 1.0
        N4 = 0.25 * A4 * B4 * C4
        dN4_dxi = -0.25 * B4 * (C4 + A4)
        dN4_deta = 0.25 * A4 * (C4 + B4)
        N5 = 0.5 * (1.0 - xi * xi) * (1.0 - eta)
        dN5_dxi = -xi * (1.0 - eta)
        dN5_deta = -0.5 * (1.0 - xi * xi)
        N6 = 0.5 * (1.0 + xi) * (1.0 - eta * eta)
        dN6_dxi = 0.5 * (1.0 - eta * eta)
        dN6_deta = -(1.0 + xi) * eta
        N7 = 0.5 * (1.0 - xi * xi) * (1.0 + eta)
        dN7_dxi = -xi * (1.0 + eta)
        dN7_deta = 0.5 * (1.0 - xi * xi)
        N8 = 0.5 * (1.0 - xi) * (1.0 - eta * eta)
        dN8_dxi = -0.5 * (1.0 - eta * eta)
        dN8_deta = -(1.0 - xi) * eta
        N = np.array([N1, N2, N3, N4, N5, N6, N7, N8], dtype=float)
        dN_dxi = np.array([dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi, dN5_dxi, dN6_dxi, dN7_dxi, dN8_dxi], dtype=float)
        dN_deta = np.array([dN1_deta, dN2_deta, dN3_deta, dN4_deta, dN5_deta, dN6_deta, dN7_deta, dN8_deta], dtype=float)
        return (N, dN_dxi, dN_deta)
    enodes = edge_nodes[face]
    for s, w in zip(gp, gw):
        if face == 0:
            xi, eta = (float(s), -1.0)
            var_xi = True
        elif face == 1:
            xi, eta = (1.0, float(s))
            var_xi = False
        elif face == 2:
            xi, eta = (float(s), 1.0)
            var_xi = True
        else:
            xi, eta = (-1.0, float(s))
            var_xi = False
        N, dN_dxi, dN_deta = shape_Q8(xi, eta)
        if var_xi:
            dN = dN_dxi
        else:
            dN = dN_deta
        tx = float(np.dot(dN, X[:, 0]))
        ty = float(np.dot(dN, X[:, 1]))
        J = (tx * tx + ty * ty) ** 0.5
        for a in enodes:
            coeff = N[a] * J * w
            r_elem[2 * a] += t[0] * coeff
            r_elem[2 * a + 1] += t[1] * coeff
    return r_elem
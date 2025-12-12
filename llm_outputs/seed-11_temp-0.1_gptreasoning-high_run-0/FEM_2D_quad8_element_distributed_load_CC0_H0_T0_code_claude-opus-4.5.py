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
    edge_nodes = {0: (0, 4, 1), 1: (1, 5, 2), 2: (2, 6, 3), 3: (3, 7, 0)}
    (n_start, n_mid, n_end) = edge_nodes[face]
    x_edge = node_coords[[n_start, n_mid, n_end], :]
    if num_gauss_pts == 1:
        gauss_pts = np.array([0.0])
        gauss_wts = np.array([2.0])
    elif num_gauss_pts == 2:
        gauss_pts = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
        gauss_wts = np.array([1.0, 1.0])
    elif num_gauss_pts == 3:
        gauss_pts = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        gauss_wts = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    r_elem = np.zeros(16)

    def shape_funcs(s):
        N1 = 0.5 * s * (s - 1.0)
        N2 = (1.0 - s) * (1.0 + s)
        N3 = 0.5 * s * (s + 1.0)
        return np.array([N1, N2, N3])

    def shape_derivs(s):
        dN1 = s - 0.5
        dN2 = -2.0 * s
        dN3 = s + 0.5
        return np.array([dN1, dN2, dN3])
    for (gp, gw) in zip(gauss_pts, gauss_wts):
        N = shape_funcs(gp)
        dN = shape_derivs(gp)
        dx_ds = np.dot(dN, x_edge[:, 0])
        dy_ds = np.dot(dN, x_edge[:, 1])
        J = np.sqrt(dx_ds ** 2 + dy_ds ** 2)
        for (i, node_idx) in enumerate([n_start, n_mid, n_end]):
            r_elem[2 * node_idx] += N[i] * traction[0] * J * gw
            r_elem[2 * node_idx + 1] += N[i] * traction[1] * J * gw
    return r_elem
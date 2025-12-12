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
    (start_idx, mid_idx, end_idx) = edge_nodes[face]
    p_start = node_coords[start_idx]
    p_mid = node_coords[mid_idx]
    p_end = node_coords[end_idx]
    if num_gauss_pts == 1:
        s_gauss = np.array([0.0])
        w_gauss = np.array([2.0])
    elif num_gauss_pts == 2:
        s_gauss = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
        w_gauss = np.array([1.0, 1.0])
    elif num_gauss_pts == 3:
        s_gauss = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        w_gauss = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    else:
        raise ValueError(f'num_gauss_pts must be 1, 2, or 3, got {num_gauss_pts}')

    def N_quad(s):
        """1D quadratic shape functions on [-1, 1] for nodes at s=-1, 0, 1"""
        N1 = 0.5 * s * (s - 1)
        N2 = (1 - s) * (1 + s)
        N3 = 0.5 * s * (s + 1)
        return np.array([N1, N2, N3])

    def dN_ds_quad(s):
        """Derivatives of 1D quadratic shape functions"""
        dN1 = s - 0.5
        dN2 = -2 * s
        dN3 = s + 0.5
        return np.array([dN1, dN2, dN3])
    r_edge = np.zeros(6)
    for (i, (s, w)) in enumerate(zip(s_gauss, w_gauss)):
        N = N_quad(s)
        dN_ds = dN_ds_quad(s)
        x = N[0] * p_start[0] + N[1] * p_mid[0] + N[2] * p_end[0]
        y = N[0] * p_start[1] + N[1] * p_mid[1] + N[2] * p_end[1]
        dx_ds = dN_ds[0] * p_start[0] + dN_ds[1] * p_mid[0] + dN_ds[2] * p_end[0]
        dy_ds = dN_ds[0] * p_start[1] + dN_ds[1] * p_mid[1] + dN_ds[2] * p_end[1]
        J = np.sqrt(dx_ds ** 2 + dy_ds ** 2)
        for j in range(3):
            r_edge[2 * j] += N[j] * traction[0] * J * w
            r_edge[2 * j + 1] += N[j] * traction[1] * J * w
    r_elem = np.zeros(16)
    node_indices = [start_idx, mid_idx, end_idx]
    for (j, node_idx) in enumerate(node_indices):
        r_elem[2 * node_idx] = r_edge[2 * j]
        r_elem[2 * node_idx + 1] = r_edge[2 * j + 1]
    return r_elem
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
    face_nodes = {0: (0, 4, 1), 1: (1, 5, 2), 2: (2, 6, 3), 3: (3, 7, 0)}
    (start_idx, mid_idx, end_idx) = face_nodes[face]
    p_start = node_coords[start_idx]
    p_mid = node_coords[mid_idx]
    p_end = node_coords[end_idx]
    if num_gauss_pts == 1:
        gauss_pts = np.array([0.0])
        gauss_wts = np.array([2.0])
    elif num_gauss_pts == 2:
        gauss_pts = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
        gauss_wts = np.array([1.0, 1.0])
    elif num_gauss_pts == 3:
        gauss_pts = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        gauss_wts = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    else:
        raise ValueError(f'num_gauss_pts must be 1, 2, or 3, got {num_gauss_pts}')

    def shape_funcs(s):
        """Q8 1D shape functions: N1, N_mid, N2"""
        N1 = -0.5 * s * (1 - s)
        N_mid = (1 - s) * (1 + s)
        N2 = 0.5 * s * (1 + s)
        return np.array([N1, N_mid, N2])

    def shape_derivs(s):
        """Derivatives of Q8 1D shape functions w.r.t. s"""
        dN1_ds = -0.5 * (1 - 2 * s)
        dN_mid_ds = -2 * s
        dN2_ds = 0.5 * (1 + 2 * s)
        return np.array([dN1_ds, dN_mid_ds, dN2_ds])
    r_edge = np.zeros(6)
    for (gp, wt) in zip(gauss_pts, gauss_wts):
        N = shape_funcs(gp)
        dN_ds = shape_derivs(gp)
        x = N[0] * p_start + N[1] * p_mid + N[2] * p_end
        dx_ds = dN_ds[0] * p_start + dN_ds[1] * p_mid + dN_ds[2] * p_end
        edge_length_elem = np.linalg.norm(dx_ds)
        for i in range(3):
            r_edge[2 * i:2 * i + 2] += N[i] * traction * edge_length_elem * wt
    r_elem = np.zeros(16)
    node_indices = [start_idx, mid_idx, end_idx]
    for (local_i, global_i) in enumerate(node_indices):
        r_elem[2 * global_i:2 * global_i + 2] = r_edge[2 * local_i:2 * local_i + 2]
    return r_elem
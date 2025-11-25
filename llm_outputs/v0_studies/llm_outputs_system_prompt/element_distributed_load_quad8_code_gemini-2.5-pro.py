def element_distributed_load_quad8(face: int, node_coords: np.ndarray, traction: np.ndarray, num_gauss_pts: int) -> np.ndarray:
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
    face_nodes_map = {0: (0, 4, 1), 1: (1, 5, 2), 2: (2, 6, 3), 3: (3, 7, 0)}
    edge_node_indices = face_nodes_map[face]
    edge_coords = node_coords[np.array(edge_node_indices)]
    N_funcs = [lambda s: 0.5 * s * (s - 1.0), lambda s: 1.0 - s ** 2, lambda s: 0.5 * s * (s + 1.0)]
    dN_ds_funcs = [lambda s: s - 0.5, lambda s: -2.0 * s, lambda s: s + 0.5]
    gauss_map = {1: (np.array([0.0]), np.array([2.0])), 2: (np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]), np.array([1.0, 1.0])), 3: (np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)]), np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]))}
    (gauss_pts, gauss_wts) = gauss_map[num_gauss_pts]
    f_edge = np.zeros(6)
    for (s, w) in zip(gauss_pts, gauss_wts):
        N_vals = np.array([f(s) for f in N_funcs])
        dN_ds_vals = np.array([f(s) for f in dN_ds_funcs])
        J_vec = np.dot(dN_ds_vals, edge_coords)
        det_J = np.linalg.norm(J_vec)
        force_increment = w * det_J * np.kron(N_vals, traction)
        f_edge += force_increment
    r_elem = np.zeros(16)
    (start_node, mid_node, end_node) = edge_node_indices
    dof_indices = np.array([2 * start_node, 2 * start_node + 1, 2 * mid_node, 2 * mid_node + 1, 2 * end_node, 2 * end_node + 1])
    r_elem[dof_indices] = f_edge
    return r_elem
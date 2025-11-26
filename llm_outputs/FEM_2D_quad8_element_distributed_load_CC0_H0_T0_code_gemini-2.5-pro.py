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
    if num_gauss_pts == 1:
        s_pts = np.array([0.0])
        w_pts = np.array([2.0])
    elif num_gauss_pts == 2:
        s_pts = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
        w_pts = np.array([1.0, 1.0])
    elif num_gauss_pts == 3:
        s_pts = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        w_pts = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    else:
        raise ValueError('num_gauss_pts must be 1, 2, or 3')
    face_nodes_map = {0: [0, 4, 1], 1: [1, 5, 2], 2: [2, 6, 3], 3: [3, 7, 0]}
    edge_node_indices = face_nodes_map[face]
    edge_node_coords = node_coords[edge_node_indices]
    N_vals = np.array([0.5 * s_pts * (s_pts - 1), 1.0 - s_pts ** 2, 0.5 * s_pts * (s_pts + 1)])
    dN_ds_vals = np.array([s_pts - 0.5, -2.0 * s_pts, s_pts + 0.5])
    dx_ds = dN_ds_vals.T @ edge_node_coords[:, 0]
    dy_ds = dN_ds_vals.T @ edge_node_coords[:, 1]
    J = np.sqrt(dx_ds ** 2 + dy_ds ** 2)
    integrand_factor = w_pts * J
    integrated_N = np.sum(N_vals * integrand_factor, axis=1)
    nodal_forces = np.outer(integrated_N, traction)
    r_elem = np.zeros(16)
    dof_indices_x = 2 * np.array(edge_node_indices)
    dof_indices_y = dof_indices_x + 1
    r_elem[dof_indices_x] = nodal_forces[:, 0]
    r_elem[dof_indices_y] = nodal_forces[:, 1]
    return r_elem
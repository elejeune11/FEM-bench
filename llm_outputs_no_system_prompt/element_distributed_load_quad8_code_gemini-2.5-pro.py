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
    if num_gauss_pts == 1:
        gauss_pts = np.array([0.0])
        gauss_wts = np.array([2.0])
    elif num_gauss_pts == 2:
        p = 1.0 / np.sqrt(3.0)
        gauss_pts = np.array([-p, p])
        gauss_wts = np.array([1.0, 1.0])
    elif num_gauss_pts == 3:
        p = np.sqrt(3.0 / 5.0)
        gauss_pts = np.array([-p, 0.0, p])
        gauss_wts = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    else:
        raise ValueError('num_gauss_pts must be 1, 2, or 3')
    edge_nodes_map = {0: (0, 4, 1), 1: (1, 5, 2), 2: (2, 6, 3), 3: (3, 7, 0)}
    if face not in edge_nodes_map:
        raise ValueError('face must be 0, 1, 2, or 3')
    (n_start, n_mid, n_end) = edge_nodes_map[face]
    coords_start = node_coords[n_start]
    coords_mid = node_coords[n_mid]
    coords_end = node_coords[n_end]
    r_elem = np.zeros(16)
    for (s, w) in zip(gauss_pts, gauss_wts):
        N_start = 0.5 * s * (s - 1.0)
        N_mid = 1.0 - s ** 2
        N_end = 0.5 * s * (s + 1.0)
        dN_start_ds = s - 0.5
        dN_mid_ds = -2.0 * s
        dN_end_ds = s + 0.5
        dx_ds = dN_start_ds * coords_start + dN_mid_ds * coords_mid + dN_end_ds * coords_end
        J_s = np.linalg.norm(dx_ds)
        r_elem[2 * n_start:2 * n_start + 2] += N_start * traction * J_s * w
        r_elem[2 * n_mid:2 * n_mid + 2] += N_mid * traction * J_s * w
        r_elem[2 * n_end:2 * n_end + 2] += N_end * traction * J_s * w
    return r_elem
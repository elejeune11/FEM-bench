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
    face_nodes = [(0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0)]
    (start_idx, mid_idx, end_idx) = face_nodes[face]
    if num_gauss_pts == 1:
        gauss_pts = np.array([0.0])
        gauss_wts = np.array([2.0])
    elif num_gauss_pts == 2:
        gauss_pts = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
        gauss_wts = np.array([1.0, 1.0])
    elif num_gauss_pts == 3:
        gauss_pts = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        gauss_wts = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    else:
        raise ValueError('num_gauss_pts must be 1, 2, or 3')
    r_edge = np.zeros(6)
    for (s, wt) in zip(gauss_pts, gauss_wts):
        N = np.array([s * (s - 1) / 2, 1 - s * s, s * (s + 1) / 2])
        dN_ds = np.array([s - 0.5, -2 * s, s + 0.5])
        x_coords = node_coords[[start_idx, mid_idx, end_idx], 0]
        y_coords = node_coords[[start_idx, mid_idx, end_idx], 1]
        dx_ds = np.dot(dN_ds, x_coords)
        dy_ds = np.dot(dN_ds, y_coords)
        jac = np.sqrt(dx_ds ** 2 + dy_ds ** 2)
        r_edge[0:2] += wt * N[0] * jac * traction
        r_edge[2:4] += wt * N[1] * jac * traction
        r_edge[4:6] += wt * N[2] * jac * traction
    r_elem = np.zeros(16)
    r_elem[2 * start_idx:2 * start_idx + 2] = r_edge[0:2]
    r_elem[2 * mid_idx:2 * mid_idx + 2] = r_edge[2:4]
    r_elem[2 * end_idx:2 * end_idx + 2] = r_edge[4:6]
    return r_elem
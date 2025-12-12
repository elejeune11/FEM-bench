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
    node_coords = np.asarray(node_coords, dtype=float)
    traction = np.asarray(traction, dtype=float)
    if node_coords.shape != (8, 2):
        raise ValueError('node_coords must be shape (8,2)')
    if traction.shape != (2,):
        raise ValueError('traction must be length 2')
    if num_gauss_pts not in (1, 2, 3):
        raise ValueError('num_gauss_pts must be 1, 2, or 3')
    faces = [(0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0)]
    if face not in (0, 1, 2, 3):
        raise ValueError('face must be 0, 1, 2, or 3')
    edge_nodes = faces[face]
    coords = node_coords[list(edge_nodes), :]
    if num_gauss_pts == 1:
        s_pts = np.array([0.0], dtype=float)
        w_pts = np.array([2.0], dtype=float)
    elif num_gauss_pts == 2:
        a = 1.0 / np.sqrt(3.0)
        s_pts = np.array([-a, a], dtype=float)
        w_pts = np.array([1.0, 1.0], dtype=float)
    else:
        a = np.sqrt(3.0 / 5.0)
        s_pts = np.array([-a, 0.0, a], dtype=float)
        w_pts = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=float)
    r_elem = np.zeros(16, dtype=float)
    for (si, wi) in zip(s_pts, w_pts):
        N1 = 0.5 * si * (si - 1.0)
        N2 = 1.0 - si * si
        N3 = 0.5 * si * (si + 1.0)
        dN1 = si - 0.5
        dN2 = -2.0 * si
        dN3 = si + 0.5
        dXds = dN1 * coords[0, :] + dN2 * coords[1, :] + dN3 * coords[2, :]
        J = np.hypot(dXds[0], dXds[1])
        Nvec = np.array([N1, N2, N3], dtype=float)
        for (k_local, node) in enumerate(edge_nodes):
            idx = 2 * node
            r_elem[idx] += Nvec[k_local] * traction[0] * J * wi
            r_elem[idx + 1] += Nvec[k_local] * traction[1] * J * wi
    return r_elem
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
    if face not in (0, 1, 2, 3):
        raise ValueError('face must be one of {0,1,2,3}')
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.shape != (8, 2):
        raise ValueError('node_coords must have shape (8,2)')
    traction = np.asarray(traction, dtype=float).reshape(-1)
    if traction.size != 2:
        raise ValueError('traction must be a length-2 array-like')
    if num_gauss_pts not in (1, 2, 3):
        raise ValueError('num_gauss_pts must be one of {1,2,3}')
    edges = [(0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0)]
    (n0, nm, n1) = edges[face]
    X0 = node_coords[n0]
    Xm = node_coords[nm]
    X1 = node_coords[n1]
    if num_gauss_pts == 1:
        gp = np.array([0.0])
        gw = np.array([2.0])
    elif num_gauss_pts == 2:
        a = 1.0 / np.sqrt(3.0)
        gp = np.array([-a, a])
        gw = np.array([1.0, 1.0])
    else:
        a = np.sqrt(3.0 / 5.0)
        gp = np.array([-a, 0.0, a])
        gw = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

    def lvals(s):
        l1 = 0.5 * s * (s - 1.0)
        l2 = 1.0 - s * s
        l3 = 0.5 * s * (s + 1.0)
        return (l1, l2, l3)

    def dlvals(s):
        dl1 = s - 0.5
        dl2 = -2.0 * s
        dl3 = s + 0.5
        return (dl1, dl2, dl3)
    r_edge = np.zeros((3, 2), dtype=float)
    for (s, w) in zip(gp, gw):
        (l1, l2, l3) = lvals(s)
        (dl1, dl2, dl3) = dlvals(s)
        dxds = dl1 * X0 + dl2 * Xm + dl3 * X1
        J = np.linalg.norm(dxds)
        factor = w * J
        if factor != 0.0:
            r_edge[0] += l1 * traction * factor
            r_edge[1] += l2 * traction * factor
            r_edge[2] += l3 * traction * factor
    r_elem = np.zeros(16, dtype=float)
    nodes = (n0, nm, n1)
    for (i_local, n) in enumerate(nodes):
        r_elem[2 * n] = r_edge[i_local, 0]
        r_elem[2 * n + 1] = r_edge[i_local, 1]
    return r_elem
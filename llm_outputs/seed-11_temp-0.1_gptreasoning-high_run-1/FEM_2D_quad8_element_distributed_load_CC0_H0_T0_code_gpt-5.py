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
    traction = np.asarray(traction, dtype=float).ravel()
    if node_coords.shape != (8, 2):
        raise ValueError('node_coords must have shape (8,2)')
    if traction.shape[0] != 2:
        raise ValueError('traction must be a length-2 vector')
    if face not in (0, 1, 2, 3):
        raise ValueError('face must be in {0,1,2,3}')
    if num_gauss_pts not in (1, 2, 3):
        raise ValueError('num_gauss_pts must be in {1,2,3}')
    face_nodes = [(0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0)]
    n1, n2, n3 = face_nodes[face]
    xe = node_coords[[n1, n2, n3], 0]
    ye = node_coords[[n1, n2, n3], 1]
    if num_gauss_pts == 1:
        gpts = np.array([0.0])
        gwts = np.array([2.0])
    elif num_gauss_pts == 2:
        a = 1.0 / np.sqrt(3.0)
        gpts = np.array([-a, a])
        gwts = np.array([1.0, 1.0])
    else:
        a = np.sqrt(3.0 / 5.0)
        gpts = np.array([-a, 0.0, a])
        gwts = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    r_elem = np.zeros(16, dtype=float)
    for s, w in zip(gpts, gwts):
        N = np.array([0.5 * s * (s - 1.0), 1.0 - s * s, 0.5 * s * (s + 1.0)], dtype=float)
        dN = np.array([s - 0.5, -2.0 * s, s + 0.5], dtype=float)
        dxds = dN.dot(xe)
        dyds = dN.dot(ye)
        jac = np.hypot(dxds, dyds)
        scale = jac * w
        if scale != 0.0:
            contrib = traction * scale
            for a, Ni in zip((n1, n2, n3), N):
                r_elem[2 * a] += Ni * contrib[0]
                r_elem[2 * a + 1] += Ni * contrib[1]
    return r_elem
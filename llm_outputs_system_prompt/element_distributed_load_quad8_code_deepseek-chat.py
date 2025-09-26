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
    face_nodes = {0: [0, 4, 1], 1: [1, 5, 2], 2: [2, 6, 3], 3: [3, 7, 0]}
    if num_gauss_pts == 1:
        gauss_pts = np.array([0.0])
        gauss_wts = np.array([2.0])
    elif num_gauss_pts == 2:
        gauss_pts = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
        gauss_wts = np.array([1.0, 1.0])
    elif num_gauss_pts == 3:
        gauss_pts = np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)])
        gauss_wts = np.array([5 / 9, 8 / 9, 5 / 9])
    else:
        raise ValueError('num_gauss_pts must be 1, 2, or 3')
    node_indices = face_nodes[face]
    edge_nodes = node_coords[node_indices]
    r_edge = np.zeros(6)

    def N1(s):
        return -0.5 * s * (1 - s)

    def N2(s):
        return 0.5 * s * (1 + s)

    def N3(s):
        return (1 - s) * (1 + s)

    def dN1ds(s):
        return -0.5 + s

    def dN2ds(s):
        return 0.5 + s

    def dN3ds(s):
        return -2 * s
    for i in range(num_gauss_pts):
        s = gauss_pts[i]
        w = gauss_wts[i]
        N = np.array([N1(s), N2(s), N3(s)])
        dNds = np.array([dN1ds(s), dN2ds(s), dN3ds(s)])
        dxds = np.dot(dNds, edge_nodes[:, 0])
        dyds = np.dot(dNds, edge_nodes[:, 1])
        J = np.sqrt(dxds ** 2 + dyds ** 2)
        r_edge[0::2] += N * traction[0] * J * w
        r_edge[1::2] += N * traction[1] * J * w
    r_elem = np.zeros(16)
    for (i, node_idx) in enumerate(node_indices):
        elem_dof_x = 2 * node_idx
        elem_dof_y = 2 * node_idx + 1
        edge_dof_x = 2 * i
        edge_dof_y = 2 * i + 1
        r_elem[elem_dof_x] = r_edge[edge_dof_x]
        r_elem[elem_dof_y] = r_edge[edge_dof_y]
    return r_elem
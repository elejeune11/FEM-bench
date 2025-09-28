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
        s_pts = np.array([0.0])
        weights = np.array([2.0])
    elif num_gauss_pts == 2:
        s_pts = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
        weights = np.array([1.0, 1.0])
    else:
        s_pts = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
        weights = np.array([5 / 9, 8 / 9, 5 / 9])
    edge_nodes = {0: [0, 4, 1], 1: [1, 5, 2], 2: [2, 6, 3], 3: [3, 7, 0]}
    nodes = edge_nodes[face]
    r_elem = np.zeros(16)

    def N(s):
        return np.array([-0.5 * s * (1 - s), 1 - s * s, 0.5 * s * (1 + s)])

    def dN_ds(s):
        return np.array([-0.5 + s, -2 * s, 0.5 + s])
    for (s, w) in zip(s_pts, weights):
        N_s = N(s)
        x = np.sum([N_s[i] * node_coords[nodes[i]] for i in range(3)], axis=0)
        dN_s = dN_ds(s)
        dx_ds = np.sum([dN_s[i] * node_coords[nodes[i]] for i in range(3)], axis=0)
        J = np.sqrt(np.sum(dx_ds ** 2))
        for i in range(3):
            node = nodes[i]
            r_elem[2 * node:2 * node + 2] += N_s[i] * traction * J * w
    return r_elem
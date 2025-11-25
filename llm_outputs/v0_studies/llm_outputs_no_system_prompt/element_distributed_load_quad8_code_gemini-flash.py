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
    r_elem = np.zeros(16)
    if face == 0:
        nodes = [0, 4, 1]
    elif face == 1:
        nodes = [1, 5, 2]
    elif face == 2:
        nodes = [2, 6, 3]
    elif face == 3:
        nodes = [3, 7, 0]
    else:
        raise ValueError('Invalid face number')
    if num_gauss_pts == 1:
        gauss_pts = [0.0]
        weights = [2.0]
    elif num_gauss_pts == 2:
        gauss_pts = [-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)]
        weights = [1.0, 1.0]
    elif num_gauss_pts == 3:
        gauss_pts = [-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)]
        weights = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
    else:
        raise ValueError('Invalid number of Gauss points')
    for (gp, w) in zip(gauss_pts, weights):
        s = gp
        N = np.array([(1 - s) * (1 - s) / 4, (1 + s) * (1 + s) / 4, (1 - s) * (1 + s) / 2])
        shape_functions = np.array([N[0], N[1], N[2]])
        coords = np.array([node_coords[nodes[0]], node_coords[nodes[1]], node_coords[nodes[2]]])
        edge_length = np.linalg.norm(coords[1] - coords[0])
        for i in range(3):
            node_index = nodes[i] * 2
            r_elem[node_index:node_index + 2] += w * edge_length * shape_functions[i] * traction
    return r_elem
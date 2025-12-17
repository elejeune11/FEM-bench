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
    import numpy as np
    if face not in (0, 1, 2, 3):
        raise ValueError('face must be one of {0,1,2,3}.')
    coords = np.asarray(node_coords, dtype=float)
    if coords.shape != (8, 2):
        raise ValueError('node_coords must be an array of shape (8,2).')
    t = np.asarray(traction, dtype=float).reshape(-1)
    if t.size != 2:
        raise ValueError('traction must be a length-2 array-like [t_x, t_y].')
    if num_gauss_pts not in (1, 2, 3):
        raise ValueError('num_gauss_pts must be one of {1,2,3}.')
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
    edges = ((0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0))
    n_start, n_mid, n_end = edges[face]
    edge_nodes = (n_start, n_mid, n_end)
    edge_coords = coords[list(edge_nodes), :]
    r_edge = np.zeros((3, 2), dtype=float)
    for s, w in zip(gp, gw):
        N1 = 0.5 * s * (s - 1.0)
        N2 = 1.0 - s * s
        N3 = 0.5 * s * (s + 1.0)
        N = np.array([N1, N2, N3])
        dN = np.array([s - 0.5, -2.0 * s, s + 0.5])
        tangent = dN @ edge_coords
        J = np.hypot(tangent[0], tangent[1])
        contrib = w * J * (N[:, None] * t[None, :])
        r_edge += contrib
    r_elem = np.zeros(16, dtype=float)
    for local_i, node_i in enumerate(edge_nodes):
        r_elem[2 * node_i:2 * node_i + 2] += r_edge[local_i, :]
    return r_elem
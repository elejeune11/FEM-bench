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
    import numpy as np
    if face not in (0, 1, 2, 3):
        raise ValueError('face must be one of {0,1,2,3}')
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.shape != (8, 2):
        raise ValueError('node_coords must have shape (8,2)')
    traction = np.asarray(traction, dtype=float).reshape(-1)
    if traction.shape != (2,):
        raise ValueError('traction must be a vector of shape (2,)')
    if num_gauss_pts not in (1, 2, 3):
        raise ValueError('num_gauss_pts must be one of {1,2,3}')
    edge_map = {0: (0, 4, 1), 1: (1, 5, 2), 2: (2, 6, 3), 3: (3, 7, 0)}
    edge_nodes = edge_map[int(face)]
    coords_edge = node_coords[np.array(edge_nodes), :]
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
    r_edge = np.zeros((3, 2), dtype=float)
    for (s, w) in zip(gp, gw):
        N = np.array([0.5 * s * (s - 1.0), 1.0 - s * s, 0.5 * s * (s + 1.0)], dtype=float)
        dNds = np.array([s - 0.5, -2.0 * s, s + 0.5], dtype=float)
        tangent = dNds @ coords_edge
        J = np.linalg.norm(tangent)
        scale = J * w
        r_edge += np.outer(N * scale, traction)
    r_elem = np.zeros(16, dtype=float)
    for (i_local, i_global) in enumerate(edge_nodes):
        r_elem[2 * i_global] += r_edge[i_local, 0]
        r_elem[2 * i_global + 1] += r_edge[i_local, 1]
    return r_elem
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
    if face == 0:
        indices = [0, 4, 1]
    elif face == 1:
        indices = [1, 5, 2]
    elif face == 2:
        indices = [2, 6, 3]
    elif face == 3:
        indices = [3, 7, 0]
    else:
        raise ValueError('Invalid face index')
    edge_coords = node_coords[indices]
    if num_gauss_pts == 1:
        pts = np.array([0.0])
        wts = np.array([2.0])
    elif num_gauss_pts == 2:
        val = 1.0 / np.sqrt(3.0)
        pts = np.array([-val, val])
        wts = np.array([1.0, 1.0])
    elif num_gauss_pts == 3:
        val = np.sqrt(0.6)
        pts = np.array([-val, 0.0, val])
        wts = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    else:
        raise ValueError('Invalid num_gauss_pts')
    r_elem = np.zeros(16)
    for (s, w) in zip(pts, wts):
        N = np.array([0.5 * s * (s - 1), 1 - s ** 2, 0.5 * s * (s + 1)])
        dNds = np.array([s - 0.5, -2 * s, s + 0.5])
        dxd = dNds @ edge_coords
        detJ = np.linalg.norm(dxd)
        weighted_traction = w * detJ * traction
        for i in range(3):
            global_idx = indices[i]
            r_elem[2 * global_idx] += N[i] * weighted_traction[0]
            r_elem[2 * global_idx + 1] += N[i] * weighted_traction[1]
    return r_elem
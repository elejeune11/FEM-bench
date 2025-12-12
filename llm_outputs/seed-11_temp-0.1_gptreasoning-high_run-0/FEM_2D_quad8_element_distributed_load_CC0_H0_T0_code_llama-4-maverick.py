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
    (gauss_points, gauss_weights) = np.polynomial.legendre.leggauss(num_gauss_pts)
    edge_nodes = [(0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0)][face]
    r_elem = np.zeros(16)
    for (i, s) in enumerate(gauss_points):
        N = np.array([0.5 * s * (s - 1), 0.5 * s * (s + 1), 1 - s ** 2])
        x = np.sum(node_coords[edge_nodes, 0] * N)
        y = np.sum(node_coords[edge_nodes, 1] * N)
        dxds = np.sum(node_coords[edge_nodes, 0] * np.array([-s + 0.5, s + 0.5, -2 * s]))
        dyds = np.sum(node_coords[edge_nodes, 1] * np.array([-s + 0.5, s + 0.5, -2 * s]))
        ds_length = np.sqrt(dxds ** 2 + dyds ** 2)
        for (j, node) in enumerate(edge_nodes):
            r_elem[2 * node] += traction[0] * N[j] * ds_length * gauss_weights[i]
            r_elem[2 * node + 1] += traction[1] * N[j] * ds_length * gauss_weights[i]
    return r_elem
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
    edge_nodes = {0: [0, 4, 1], 1: [1, 5, 2], 2: [2, 6, 3], 3: [3, 7, 0]}
    nodes = edge_nodes[face]
    edge_length = np.linalg.norm(node_coords[nodes[2]] - node_coords[nodes[0]])
    if num_gauss_pts == 1:
        gauss_pts = [0]
        gauss_weights = [2]
    elif num_gauss_pts == 2:
        gauss_pts = [-np.sqrt(1 / 3), np.sqrt(1 / 3)]
        gauss_weights = [1, 1]
    elif num_gauss_pts == 3:
        gauss_pts = [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)]
        gauss_weights = [5 / 9, 8 / 9, 5 / 9]
    r_elem = np.zeros(16)
    for (pt, weight) in zip(gauss_pts, gauss_weights):
        x_gp = node_coords[nodes[0]] * (1 - pt) / 2 + node_coords[nodes[1]] * (1 + pt) / 2
        jacobian = edge_length / 2
        load_contribution = traction * weight * jacobian
        r_elem[nodes[0] * 2] += load_contribution[0] * (1 - pt) / 2
        r_elem[nodes[0] * 2 + 1] += load_contribution[1] * (1 - pt) / 2
        r_elem[nodes[1] * 2] += load_contribution[0] * (1 + pt) / 2
        r_elem[nodes[1] * 2 + 1] += load_contribution[1] * (1 + pt) / 2
        r_elem[nodes[2] * 2] += 0
        r_elem[nodes[2] * 2 + 1] += 0
    r_elem[nodes[0] * 2] += traction[0] * edge_length / 2
    r_elem[nodes[0] * 2 + 1] += traction[1] * edge_length / 2
    r_elem[nodes[2] * 2] += traction[0] * edge_length / 2
    r_elem[nodes[2] * 2 + 1] += traction[1] * edge_length / 2
    r_elem[nodes[1] * 2] += traction[0] * edge_length / 2
    r_elem[nodes[1] * 2 + 1] += traction[1] * edge_length / 2
    return r_elem
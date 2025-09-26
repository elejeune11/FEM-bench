def assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements):
    """
    Assembles the global stiffness matrix for a 3D linear elastic frame structure composed of beam elements.
    Each beam element connects two nodes and contributes a 12x12 stiffness matrix (6 DOFs per node) to the
    global stiffness matrix. The element stiffness is computed in the local coordinate system using material
    and geometric properties, then transformed into the global coordinate system via a transformation matrix.
    Parameters
    ----------
    node_coords : ndarray of shape (n_nodes, 3)
        Array containing the (x, y, z) coordinates of each node.
    elements : list of dict
        A list of element dictionaries. Each dictionary must contain:
                Indices of the start and end nodes.
                Young's modulus of the element.
                Poisson's ratio of the element.
                Cross-sectional area.
                Second moments of area about local y and z axes.
                Torsional constant.
                Optional vector defining the local z-direction to resolve transformation ambiguity.
    Returns
    -------
    K : ndarray of shape (6 * n_nodes, 6 * n_nodes)
        The assembled global stiffness matrix of the structure, with 6 degrees of freedom per node.
    Helper Functions Required
    -------------------------
        Returns the 12x12 local stiffness matrix for a 3D beam element.
        Returns the 12x12 transformation matrix to convert local stiffness to global coordinates.
    Notes
    -----
    """
    import numpy as np
    n_nodes = int(np.asarray(node_coords).shape[0])
    K = np.zeros((6 * n_nodes, 6 * n_nodes), dtype=float)
    coords = np.asarray(node_coords, dtype=float)
    for elem in elements:
        i = int(elem['node_i'])
        j = int(elem['node_j'])
        (xi, yi, zi) = coords[i]
        (xj, yj, zj) = coords[j]
        dx = xj - xi
        dy = yj - yi
        dz = zj - zi
        L = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        Kl = local_elastic_stiffness_matrix_3D_beam(float(elem['E']), float(elem['nu']), float(elem['A']), L, float(elem['I_y']), float(elem['I_z']), float(elem['J']))
        ref_vec = elem.get('local_z', None)
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ref_vec=None if ref_vec is None else np.asarray(ref_vec, dtype=float))
        Ke = Gamma.T @ Kl @ Gamma
        dofs_i = np.arange(6 * i, 6 * i + 6)
        dofs_j = np.arange(6 * j, 6 * j + 6)
        dofs = np.concatenate((dofs_i, dofs_j))
        K[np.ix_(dofs, dofs)] += Ke
    return K
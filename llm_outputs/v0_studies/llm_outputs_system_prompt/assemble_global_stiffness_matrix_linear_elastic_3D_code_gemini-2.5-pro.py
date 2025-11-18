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
    n_nodes = node_coords.shape[0]
    n_dofs = 6 * n_nodes
    K = np.zeros((n_dofs, n_dofs))
    for element in elements:
        node_i_idx = element['node_i']
        node_j_idx = element['node_j']
        coords_i = node_coords[node_i_idx]
        coords_j = node_coords[node_j_idx]
        (xi, yi, zi) = coords_i
        (xj, yj, zj) = coords_j
        L = np.linalg.norm(coords_j - coords_i)
        E = element['E']
        nu = element['nu']
        A = element['A']
        Iy = element['I_y']
        Iz = element['I_z']
        J = element['J']
        ref_vec = element.get('local_z', None)
        k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ref_vec=ref_vec)
        k_global = Gamma.T @ k_local @ Gamma
        dofs_i = np.arange(6 * node_i_idx, 6 * node_i_idx + 6)
        dofs_j = np.arange(6 * node_j_idx, 6 * node_j_idx + 6)
        global_dofs = np.concatenate((dofs_i, dofs_j))
        K[np.ix_(global_dofs, global_dofs)] += k_global
    return K
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
    K = np.zeros((6 * n_nodes, 6 * n_nodes))
    for element in elements:
        i = element['node_i']
        j = element['node_j']
        coords_i = node_coords[i]
        coords_j = node_coords[j]
        L = np.linalg.norm(coords_j - coords_i)
        local_z = element.get('local_z', None)
        if local_z is not None:
            local_z = np.array(local_z)
        Gamma = beam_transformation_matrix_3D(coords_i[0], coords_i[1], coords_i[2], coords_j[0], coords_j[1], coords_j[2], local_z)
        k_local = local_elastic_stiffness_matrix_3D_beam(element['E'], element['nu'], element['A'], L, element['I_y'], element['I_z'], element['J'])
        k_global = Gamma.T @ k_local @ Gamma
        dofs_i = slice(6 * i, 6 * i + 6)
        dofs_j = slice(6 * j, 6 * j + 6)
        K[dofs_i, dofs_i] += k_global[0:6, 0:6]
        K[dofs_i, dofs_j] += k_global[0:6, 6:12]
        K[dofs_j, dofs_i] += k_global[6:12, 0:6]
        K[dofs_j, dofs_j] += k_global[6:12, 6:12]
    return K
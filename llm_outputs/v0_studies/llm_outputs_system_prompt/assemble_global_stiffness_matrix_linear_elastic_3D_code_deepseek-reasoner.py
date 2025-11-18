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
    K_global = np.zeros((6 * n_nodes, 6 * n_nodes))
    for element in elements:
        i = element['node_i']
        j = element['node_j']
        (xi, yi, zi) = node_coords[i]
        (xj, yj, zj) = node_coords[j]
        L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
        local_z = element.get('local_z', None)
        if local_z is not None:
            local_z = np.asarray(local_z)
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z)
        K_local = local_elastic_stiffness_matrix_3D_beam(element['E'], element['nu'], element['A'], L, element['I_y'], element['I_z'], element['J'])
        K_global_element = Gamma.T @ K_local @ Gamma
        dofs_i = slice(6 * i, 6 * i + 6)
        dofs_j = slice(6 * j, 6 * j + 6)
        K_global[dofs_i, dofs_i] += K_global_element[0:6, 0:6]
        K_global[dofs_i, dofs_j] += K_global_element[0:6, 6:12]
        K_global[dofs_j, dofs_i] += K_global_element[6:12, 0:6]
        K_global[dofs_j, dofs_j] += K_global_element[6:12, 6:12]
    return K_global
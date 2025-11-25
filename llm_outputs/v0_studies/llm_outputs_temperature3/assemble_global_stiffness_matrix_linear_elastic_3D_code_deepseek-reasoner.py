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
    n_dof = 6 * n_nodes
    K = np.zeros((n_dof, n_dof))
    for element in elements:
        i = element['node_i']
        j = element['node_j']
        coords_i = node_coords[i]
        coords_j = node_coords[j]
        L = np.linalg.norm(coords_j - coords_i)
        ref_vec = element.get('local_z', None)
        K_local = local_elastic_stiffness_matrix_3D_beam(element['E'], element['nu'], element['A'], L, element['I_y'], element['I_z'], element['J'])
        Gamma = beam_transformation_matrix_3D(coords_i[0], coords_i[1], coords_i[2], coords_j[0], coords_j[1], coords_j[2], ref_vec)
        K_element_global = Gamma.T @ K_local @ Gamma
        dof_i = slice(6 * i, 6 * i + 6)
        dof_j = slice(6 * j, 6 * j + 6)
        K[dof_i, dof_i] += K_element_global[0:6, 0:6]
        K[dof_i, dof_j] += K_element_global[0:6, 6:12]
        K[dof_j, dof_i] += K_element_global[6:12, 0:6]
        K[dof_j, dof_j] += K_element_global[6:12, 6:12]
    return K
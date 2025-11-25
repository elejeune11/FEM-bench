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
    total_dofs = 6 * n_nodes
    K = np.zeros((total_dofs, total_dofs))
    for element in elements:
        node_i = element['node_i']
        node_j = element['node_j']
        coords_i = node_coords[node_i]
        coords_j = node_coords[node_j]
        (x1, y1, z1) = coords_i
        (x2, y2, z2) = coords_j
        L = np.linalg.norm(coords_j - coords_i)
        k_local = local_elastic_stiffness_matrix_3D_beam(E=element['E'], nu=element['nu'], A=element['A'], L=L, Iy=element['I_y'], Iz=element['I_z'], J=element['J'])
        ref_vec = element.get('local_z', None)
        Gamma = beam_transformation_matrix_3D(x1, y1, z1, x2, y2, z2, ref_vec)
        k_global = Gamma.T @ k_local @ Gamma
        dof_indices = np.r_[6 * node_i:6 * node_i + 6, 6 * node_j:6 * node_j + 6]
        K[np.ix_(dof_indices, dof_indices)] += k_global
    return K
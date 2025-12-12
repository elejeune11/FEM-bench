def MSA_3D_assemble_global_linear_elastic_stiffness_CC0_H2_T1(node_coords, elements):
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
        node_i = element['node_i']
        node_j = element['node_j']
        (x1, y1, z1) = node_coords[node_i]
        (x2, y2, z2) = node_coords[node_j]
        (dx, dy, dz) = (x2 - x1, y2 - y1, z2 - z1)
        L = np.sqrt(dx * dx + dy * dy + dz * dz)
        E = element['E']
        nu = element['nu']
        A = element['A']
        Iy = element['I_y']
        Iz = element['I_z']
        J = element['J']
        ref_vec = element.get('local_z', None)
        k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        gamma = beam_transformation_matrix_3D(x1, y1, z1, x2, y2, z2, ref_vec)
        k_global = gamma.T @ k_local @ gamma
        dof_indices = np.concatenate([np.arange(node_i * 6, node_i * 6 + 6), np.arange(node_j * 6, node_j * 6 + 6)])
        for (i, dof_i) in enumerate(dof_indices):
            for (j, dof_j) in enumerate(dof_indices):
                K[dof_i, dof_j] += k_global[i, j]
    return K
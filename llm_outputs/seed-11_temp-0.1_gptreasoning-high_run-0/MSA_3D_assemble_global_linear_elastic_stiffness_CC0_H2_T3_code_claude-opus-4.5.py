def MSA_3D_assemble_global_linear_elastic_stiffness_CC0_H2_T3(node_coords, elements):
    """
    Assemble the global stiffness matrix for a 3D linear-elastic frame structure composed of beam elements.
    Each 2-node Euler–Bernoulli beam contributes a 12×12 local elastic stiffness matrix
    based on its material and geometric properties (E, ν, A, I_y, I_z, J, L).
    The local matrix is expressed in the element's coordinate system and includes
    axial, bending, and torsional stiffness terms.
    For global assembly, each local stiffness is transformed into global coordinates
    using a 12×12 direction-cosine transformation matrix (Γ) derived from the element's
    start and end node coordinates and, if provided, an orientation vector (local_z).
    The global stiffness matrix K is formed by adding these transformed
    element matrices into the appropriate degree-of-freedom positions.
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
                Unit vector in global coordinates defining the local z-axis orientation.
                Must not be parallel to the beam axis. If None, a default is chosen.
                Default local_z = global z, unless the beam lies along global z — then default local_z = global y.
    Returns
    -------
    K : ndarray of shape (6 * n_nodes, 6 * n_nodes)
        The assembled global stiffness matrix of the structure, with 6 degrees of freedom per node.
    Notes
    -----
    """
    n_nodes = node_coords.shape[0]
    n_dofs = 6 * n_nodes
    K = np.zeros((n_dofs, n_dofs))
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        local_z = elem.get('local_z', None)
        coord_i = node_coords[node_i]
        coord_j = node_coords[node_j]
        dx = coord_j - coord_i
        L = np.linalg.norm(dx)
        local_x = dx / L
        if local_z is None:
            global_z = np.array([0.0, 0.0, 1.0])
            global_y = np.array([0.0, 1.0, 0.0])
            if np.abs(np.abs(np.dot(local_x, global_z)) - 1.0) < 1e-06:
                local_z = global_y
            else:
                local_z = global_z
        else:
            local_z = np.array(local_z)
        local_y = np.cross(local_z, local_x)
        local_y = local_y / np.linalg.norm(local_y)
        local_z = np.cross(local_x, local_y)
        local_z = local_z / np.linalg.norm(local_z)
        R = np.array([local_x, local_y, local_z])
        G = E / (2.0 * (1.0 + nu))
        K_local = np.zeros((12, 12))
        EA_L = E * A / L
        K_local[0, 0] = EA_L
        K_local[0, 6] = -EA_L
        K_local[6, 0] = -EA_L
        K_local[6, 6] = EA_L
        GJ_L = G * J / L
        K_local[3, 3] = GJ_L
        K_local[3, 9] = -GJ_L
        K_local[9, 3] = -GJ_L
        K_local[9, 9] = GJ_L
        EIz_L3 = E * I_z / L ** 3
        K_local[1, 1] = 12 * EIz_L3
        K_local[1, 5] = 6 * EIz_L3 * L
        K_local[1, 7] = -12 * EIz_L3
        K_local[1, 11] = 6 * EIz_L3 * L
        K_local[5, 1] = 6 * EIz_L3 * L
        K_local[5, 5] = 4 * EIz_L3 * L * L
        K_local[5, 7] = -6 * EIz_L3 * L
        K_local[5, 11] = 2 * EIz_L3 * L * L
        K_local[7, 1] = -12 * EIz_L3
        K_local[7, 5] = -6 * EIz_L3 * L
        K_local[7, 7] = 12 * EIz_L3
        K_local[7, 11] = -6 * EIz_L3 * L
        K_local[11, 1] = 6 * EIz_L3 * L
        K_local[11, 5] = 2 * EIz_L3 * L * L
        K_local[11, 7] = -6 * EIz_L3 * L
        K_local[11, 11] = 4 * EIz_L3 * L * L
        EIy_L3 = E * I_y / L ** 3
        K_local[2, 2] = 12 * EIy_L3
        K_local[2, 4] = -6 * EIy_L3 * L
        K_local[2, 8] = -12 * EIy_L3
        K_local[2, 10] = -6 * EIy_L3 * L
        K_local[4, 2] = -6 * EIy_L3 * L
        K_local[4, 4] = 4 * EIy_L3 * L * L
        K_local[4, 8] = 6 * EIy_L3 * L
        K_local[4, 10] = 2 * EIy_L3 * L * L
        K_local[8, 2] = -12 * EIy_L3
        K_local[8, 4] = 6 * EIy_L3 * L
        K_local[8, 8] = 12 * EIy_L3
        K_local[8, 10] = 6 * EIy_L3 * L
        K_local[10, 2] = -6 * EIy_L3 * L
        K_local[10, 4] = 2 * EIy_L3 * L * L
        K_local[10, 8] = 6 * EIy_L3 * L
        K_local[10, 10] = 4 * EIy_L3 * L * L
        Gamma = np.zeros((12, 12))
        Gamma[0:3, 0:3] = R
        Gamma[3:6, 3:6] = R
        Gamma[6:9, 6:9] = R
        Gamma[9:12, 9:12] = R
        K_global_elem = Gamma.T @ K_local @ Gamma
        dof_i = np.arange(6 * node_i, 6 * node_i + 6)
        dof_j = np.arange(6 * node_j, 6 * node_j + 6)
        dofs = np.concatenate([dof_i, dof_j])
        for ii in range(12):
            for jj in range(12):
                K[dofs[ii], dofs[jj]] += K_global_elem[ii, jj]
    return K
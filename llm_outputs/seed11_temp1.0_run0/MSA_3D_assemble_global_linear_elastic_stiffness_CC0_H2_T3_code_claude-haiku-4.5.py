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
    n_dof = 6 * n_nodes
    K = np.zeros((n_dof, n_dof))
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        coord_i = node_coords[node_i]
        coord_j = node_coords[node_j]
        beam_vec = coord_j - coord_i
        L = np.linalg.norm(beam_vec)
        local_x = beam_vec / L
        if 'local_z' in elem and elem['local_z'] is not None:
            local_z = np.array(elem['local_z']) / np.linalg.norm(elem['local_z'])
        elif abs(local_x[2]) < 0.9:
            local_z = np.array([0.0, 0.0, 1.0])
        else:
            local_z = np.array([0.0, 1.0, 0.0])
        local_y = np.cross(local_z, local_x)
        local_y = local_y / np.linalg.norm(local_y)
        local_z = np.cross(local_x, local_y)
        local_z = local_z / np.linalg.norm(local_z)
        R = np.array([local_x, local_y, local_z]).T
        Gamma = np.zeros((12, 12))
        for i in range(2):
            Gamma[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
            Gamma[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
        G = E / (2 * (1 + nu))
        K_local = np.zeros((12, 12))
        k_a = E * A / L
        K_local[0, 0] = k_a
        K_local[0, 6] = -k_a
        K_local[6, 0] = -k_a
        K_local[6, 6] = k_a
        k_t = G * J / L
        K_local[3, 3] = k_t
        K_local[3, 9] = -k_t
        K_local[9, 3] = -k_t
        K_local[9, 9] = k_t
        k_by_1 = 12 * E * I_z / L ** 3
        k_by_2 = 6 * E * I_z / L ** 2
        k_by_3 = 4 * E * I_z / L
        k_by_4 = 2 * E * I_z / L
        K_local[1, 1] = k_by_1
        K_local[1, 5] = k_by_2
        K_local[1, 7] = -k_by_1
        K_local[1, 11] = k_by_2
        K_local[5, 1] = k_by_2
        K_local[5, 5] = k_by_3
        K_local[5, 7] = -k_by_2
        K_local[5, 11] = k_by_4
        K_local[7, 1] = -k_by_1
        K_local[7, 5] = -k_by_2
        K_local[7, 7] = k_by_1
        K_local[7, 11] = -k_by_2
        K_local[11, 1] = k_by_2
        K_local[11, 5] = k_by_4
        K_local[11, 7] = -k_by_2
        K_local[11, 11] = k_by_3
        k_bz_1 = 12 * E * I_y / L ** 3
        k_bz_2 = 6 * E * I_y / L ** 2
        k_bz_3 = 4 * E * I_y / L
        k_bz_4 = 2 * E * I_y / L
        K_local[2, 2] = k_bz_1
        K_local[2, 4] = -k_bz_2
        K_local[2, 8] = -k_bz_1
        K_local[2, 10] = -k_bz_2
        K_local[4, 2] = -k_bz_2
        K_local[4, 4] = k_bz_3
        K_local[4, 8] = k_bz_2
        K_local[4, 10] = k_bz_4
        K_local[8, 2] = -k_bz_1
        K_local[8, 4] = k_bz_2
        K_local[8, 8] = k_bz_1
        K_local[8, 10] = k_bz_2
        K_local[10, 2] = -k_bz_2
        K_local[10, 4] = k_bz_4
        K_local[10, 8] = k_bz_2
        K_local[10, 10] = k_bz_3
        K_global = Gamma.T @ K_local @ Gamma
        dof_i = np.arange(6 * node_i, 6 * node_i + 6)
        dof_j = np.arange(6 * node_j, 6 * node_j + 6)
        dof_indices = np.concatenate([dof_i, dof_j])
        for (local_row, global_row) in enumerate(dof_indices):
            for (local_col, global_col) in enumerate(dof_indices):
                K[global_row, global_col] += K_global[local_row, local_col]
    return K
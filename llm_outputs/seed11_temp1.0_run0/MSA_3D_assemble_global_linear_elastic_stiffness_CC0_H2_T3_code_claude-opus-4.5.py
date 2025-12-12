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
        global_z = np.array([0.0, 0.0, 1.0])
        global_y = np.array([0.0, 1.0, 0.0])
        if local_z is None:
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
        Gamma = np.zeros((12, 12))
        for i in range(4):
            Gamma[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        G = E / (2 * (1 + nu))
        K_local = np.zeros((12, 12))
        K_local[0, 0] = E * A / L
        K_local[0, 6] = -E * A / L
        K_local[6, 0] = -E * A / L
        K_local[6, 6] = E * A / L
        K_local[3, 3] = G * J / L
        K_local[3, 9] = -G * J / L
        K_local[9, 3] = -G * J / L
        K_local[9, 9] = G * J / L
        k1 = 12 * E * I_z / L ** 3
        k2 = 6 * E * I_z / L ** 2
        k3 = 4 * E * I_z / L
        k4 = 2 * E * I_z / L
        K_local[1, 1] = k1
        K_local[1, 5] = k2
        K_local[1, 7] = -k1
        K_local[1, 11] = k2
        K_local[5, 1] = k2
        K_local[5, 5] = k3
        K_local[5, 7] = -k2
        K_local[5, 11] = k4
        K_local[7, 1] = -k1
        K_local[7, 5] = -k2
        K_local[7, 7] = k1
        K_local[7, 11] = -k2
        K_local[11, 1] = k2
        K_local[11, 5] = k4
        K_local[11, 7] = -k2
        K_local[11, 11] = k3
        k5 = 12 * E * I_y / L ** 3
        k6 = 6 * E * I_y / L ** 2
        k7 = 4 * E * I_y / L
        k8 = 2 * E * I_y / L
        K_local[2, 2] = k5
        K_local[2, 4] = -k6
        K_local[2, 8] = -k5
        K_local[2, 10] = -k6
        K_local[4, 2] = -k6
        K_local[4, 4] = k7
        K_local[4, 8] = k6
        K_local[4, 10] = k8
        K_local[8, 2] = -k5
        K_local[8, 4] = k6
        K_local[8, 8] = k5
        K_local[8, 10] = k6
        K_local[10, 2] = -k6
        K_local[10, 4] = k8
        K_local[10, 8] = k6
        K_local[10, 10] = k7
        K_global_elem = Gamma.T @ K_local @ Gamma
        dofs_i = list(range(6 * node_i, 6 * node_i + 6))
        dofs_j = list(range(6 * node_j, 6 * node_j + 6))
        dofs = dofs_i + dofs_j
        for (ii, di) in enumerate(dofs):
            for (jj, dj) in enumerate(dofs):
                K[di, dj] += K_global_elem[ii, jj]
    return K
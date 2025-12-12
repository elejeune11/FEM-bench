def MSA_3D_assemble_global_linear_elastic_stiffness_CC0_H2_T3(node_coords, elements):
    """
    Assemble the global stiffness matrix for a 3D linear-elastic frame structure composed of beam elements.
    Each 2-node Euler–Bernoulli beam contributes a 12×12 local elastic stiffness matrix
    based on its material and geometric properties (E, ν, A, I_y, I_z, J, L).
    The local matrix is expressed in the element’s coordinate system and includes
    axial, bending, and torsional stiffness terms.
    For global assembly, each local stiffness is transformed into global coordinates
    using a 12×12 direction-cosine transformation matrix (Γ) derived from the element’s
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
    n_nodes = len(node_coords)
    K = np.zeros((6 * n_nodes, 6 * n_nodes))
    for element in elements:
        (node_i, node_j) = (element['node_i'], element['node_j'])
        (E, nu) = (element['E'], element['nu'])
        (A, I_y, I_z, J) = (element['A'], element['I_y'], element['I_z'], element['J'])
        local_z = element.get('local_z', None)
        (coord_i, coord_j) = (node_coords[node_i], node_coords[node_j])
        L = np.linalg.norm(coord_j - coord_i)
        direction_x = (coord_j - coord_i) / L
        if local_z is None:
            if np.allclose(direction_x, [0, 0, 1]) or np.allclose(direction_x, [0, 0, -1]):
                local_z = np.array([0, 1, 0])
            else:
                local_z = np.array([0, 0, 1])
        local_z = np.array(local_z) / np.linalg.norm(local_z)
        if np.allclose(np.abs(np.dot(direction_x, local_z)), 1):
            raise ValueError('local_z must not be parallel to the beam axis')
        direction_y = np.cross(direction_x, local_z)
        direction_y = direction_y / np.linalg.norm(direction_y)
        direction_z = np.cross(direction_x, direction_y)
        Gamma = np.eye(4)
        Gamma[:3, :3] = np.column_stack((direction_x, direction_y, direction_z))
        Gamma = np.block([[Gamma, np.zeros((4, 8))], [np.zeros((4, 4)), Gamma, np.zeros((4, 4))], [np.zeros((4, 8)), Gamma]])
        k_local = np.zeros((12, 12))
        k_local[0, 0] = E * A / L
        k_local[6, 6] = E * A / L
        k_local[0, 6] = -E * A / L
        k_local[6, 0] = -E * A / L
        k_local[3, 3] = E * J / (2 * (1 + nu) * L)
        k_local[9, 9] = E * J / (2 * (1 + nu) * L)
        k_local[3, 9] = -E * J / (2 * (1 + nu) * L)
        k_local[9, 3] = -E * J / (2 * (1 + nu) * L)
        k_local[1, 1] = 12 * E * I_z / L ** 3
        k_local[1, 5] = 6 * E * I_z / L ** 2
        k_local[1, 7] = -12 * E * I_z / L ** 3
        k_local[1, 11] = 6 * E * I_z / L ** 2
        k_local[5, 1] = 6 * E * I_z / L ** 2
        k_local[5, 5] = 4 * E * I_z / L
        k_local[5, 7] = -6 * E * I_z / L ** 2
        k_local[5, 11] = 2 * E * I_z / L
        k_local[7, 1] = -12 * E * I_z / L ** 3
        k_local[7, 5] = -6 * E * I_z / L ** 2
        k_local[7, 7] = 12 * E * I_z / L ** 3
        k_local[7, 11] = -6 * E * I_z / L ** 2
        k_local[11, 1] = 6 * E * I_z / L ** 2
        k_local[11, 5] = 2 * E * I_z / L
        k_local[11, 7] = -6 * E * I_z / L ** 2
        k_local[11, 11] = 4 * E * I_z / L
        k_local[2, 2] = 12 * E * I_y / L ** 3
        k_local[2, 4] = -6 * E * I_y / L ** 2
        k_local[2, 8] = -12 * E * I_y / L ** 3
        k_local[2, 10] = -6 * E * I_y / L ** 2
        k_local[4, 2] = -6 * E * I_y / L ** 2
        k_local[4, 4] = 4 * E * I_y / L
        k_local[4, 8] = 6 * E * I_y / L ** 2
        k_local[4, 10] = 2 * E * I_y / L
        k_local[8, 2] = -12 * E * I_y / L ** 3
        k_local[8, 4] = 6 * E * I_y / L ** 2
        k_local[8, 8] = 12 * E * I_y / L ** 3
        k_local[8, 10] = 6 * E * I_y / L ** 2
        k_local[10, 2] = -6 * E * I_y / L ** 2
        k_local[10, 4] = 2 * E * I_y / L
        k_local[10, 8] = 6 * E * I_y / L ** 2
        k_local[10, 10] = 4 * E * I_y / L
        k_global = np.dot(np.dot(Gamma.T, k_local), Gamma)
        dofs_i = [6 * node_i, 6 * node_i + 1, 6 * node_i + 2, 6 * node_i + 3, 6 * node_i + 4, 6 * node_i + 5]
        dofs_j = [6 * node_j, 6 * node_j + 1, 6 * node_j + 2, 6 * node_j + 3, 6 * node_j + 4, 6 * node_j + 5]
        dofs = dofs_i + dofs_j
        for (i, dof_i) in enumerate(dofs):
            for (j, dof_j) in enumerate(dofs):
                K[dof_i, dof_j] += k_global[i, j]
    return K
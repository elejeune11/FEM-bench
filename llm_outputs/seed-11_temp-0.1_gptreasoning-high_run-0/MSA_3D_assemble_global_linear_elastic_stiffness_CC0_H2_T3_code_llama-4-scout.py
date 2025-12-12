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
        (E, nu, A, I_y, I_z, J) = (element['E'], element['nu'], element['A'], element['I_y'], element['I_z'], element['J'])
        local_z = element.get('local_z')
        L = np.linalg.norm(node_coords[node_j] - node_coords[node_i])
        k_l = np.array([[E * A / L, 0, 0, 0, 0, 0, -E * A / L, 0, 0, 0, 0, 0], [0, 12 * E * I_z / L ** 3, 0, 0, 0, 6 * E * I_z / L ** 2, 0, -12 * E * I_z / L ** 3, 0, 0, 0, 6 * E * I_z / L ** 2], [0, 0, 12 * E * I_y / L ** 3, 0, -6 * E * I_y / L ** 2, 0, 0, 0, -12 * E * I_y / L ** 3, 0, -6 * E * I_y / L ** 2, 0], [0, 0, 0, G * J / L, 0, 0, 0, 0, 0, G * J / L, 0, 0], [0, 0, -6 * E * I_y / L ** 2, 0, 4 * E * I_y / L, 0, 0, 0, 6 * E * I_y / L ** 2, 0, 2 * E * I_y / L, 0], [0, 6 * E * I_z / L ** 2, 0, 0, 0, 4 * E * I_z / L, 0, -6 * E * I_z / L ** 2, 0, 0, 0, 2 * E * I_z / L], [-E * A / L, 0, 0, 0, 0, 0, E * A / L, 0, 0, 0, 0, 0], [0, -12 * E * I_z / L ** 3, 0, 0, 0, -6 * E * I_z / L ** 2, 0, 12 * E * I_z / L ** 3, 0, 0, 0, -6 * E * I_z / L ** 2], [0, 0, -12 * E * I_y / L ** 3, 0, 6 * E * I_y / L ** 2, 0, 0, 0, 12 * E * I_y / L ** 3, 0, 6 * E * I_y / L ** 2, 0], [0, 0, 0, G * J / L, 0, 0, 0, 0, 0, -G * J / L, 0, 0], [0, 0, -6 * E * I_y / L ** 2, 0, 2 * E * I_y / L, 0, 0, 0, 6 * E * I_y / L ** 2, 0, 4 * E * I_y / L, 0], [0, 6 * E * I_z / L ** 2, 0, 0, 0, 2 * E * I_z / L, 0, -6 * E * I_z / L ** 2, 0, 0, 0, 4 * E * I_z / L]])
        beam_axis = (node_coords[node_j] - node_coords[node_i]) / L
        if local_z is None:
            if np.isclose(beam_axis[0], 0) and np.isclose(beam_axis[1], 0):
                local_z = np.array([0, 1, 0])
            else:
                local_z = np.array([0, 0, 1])
        local_z = local_z / np.linalg.norm(local_z)
        local_x = beam_axis
        local_y = np.cross(local_z, local_x)
        local_y = local_y / np.linalg.norm(local_y)
        local_z = np.cross(local_x, local_y)
        T = np.array([[local_x[0], local_y[0], local_z[0], 0, 0, 0], [local_x[1], local_y[1], local_z[1], 0, 0, 0], [local_x[2], local_y[2], local_z[2], 0, 0, 0], [0, 0, 0, local_x[0], local_y[0], local_z[0]], [0, 0, 0, local_x[1], local_y[1], local_z[1]], [0, 0, 0, local_x[2], local_y[2], local_z[2]], [local_x[0], local_y[0], local_z[0], 0, 0, 0], [local_x[1], local_y[1], local_z[1], 0, 0, 0], [local_x[2], local_y[2], local_z[2], 0, 0, 0], [0, 0, 0, local_x[0], local_y[0], local_z[0]], [0, 0, 0, local_x[1], local_y[1], local_z[1]], [0, 0, 0, local_x[2], local_y[2], local_z[2]]])
        k_g = T.T @ k_l @ T
        dofs_i = np.arange(6 * node_i, 6 * node_i + 6)
        dofs_j = np.arange(6 * node_j, 6 * node_j + 6)
        K[np.ix_(dofs_i, dofs_i)] += k_g[:6, :6]
        K[np.ix_(dofs_i, dofs_j)] += k_g[:6, 6:]
        K[np.ix_(dofs_j, dofs_i)] += k_g[6:, :6]
        K[np.ix_(dofs_j, dofs_j)] += k_g[6:, 6:]
    return K
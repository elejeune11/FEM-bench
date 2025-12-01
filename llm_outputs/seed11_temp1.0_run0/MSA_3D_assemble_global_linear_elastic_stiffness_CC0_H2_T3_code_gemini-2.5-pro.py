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
                Optional vector defining the local z-direction to resolve transformation ambiguity.
    Returns
    -------
    K : ndarray of shape (6 * n_nodes, 6 * n_nodes)
        The assembled global stiffness matrix of the structure, with 6 degrees of freedom per node.
    Notes
    -----
    """
    n_nodes = node_coords.shape[0]
    K = np.zeros((6 * n_nodes, 6 * n_nodes))
    for element in elements:
        (i, j) = (element['node_i'], element['node_j'])
        (p_i, p_j) = (node_coords[i], node_coords[j])
        (E, nu, A) = (element['E'], element['nu'], element['A'])
        (I_y, I_z, J) = (element['I_y'], element['I_z'], element['J'])
        vec = p_j - p_i
        L = np.linalg.norm(vec)
        if np.isclose(L, 0.0):
            continue
        G = E / (2 * (1 + nu))
        k_local = np.zeros((12, 12))
        k_local[0, 0] = k_local[6, 6] = E * A / L
        k_local[0, 6] = k_local[6, 0] = -E * A / L
        k_local[3, 3] = k_local[9, 9] = G * J / L
        k_local[3, 9] = k_local[9, 3] = -G * J / L
        c1 = E * I_z / L ** 3
        k_bending_xy = c1 * np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L ** 2, -6 * L, 2 * L ** 2], [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L ** 2, -6 * L, 4 * L ** 2]])
        dofs_xy = np.ix_([1, 5, 7, 11], [1, 5, 7, 11])
        k_local[dofs_xy] = k_bending_xy
        c2 = E * I_y / L ** 3
        k_bending_xz = c2 * np.array([[12, -6 * L, -12, -6 * L], [-6 * L, 4 * L ** 2, 6 * L, 2 * L ** 2], [-12, 6 * L, 12, 6 * L], [-6 * L, 2 * L ** 2, 6 * L, 4 * L ** 2]])
        dofs_xz = np.ix_([2, 4, 8, 10], [2, 4, 8, 10])
        k_local[dofs_xz] = k_bending_xz
        x_prime = vec / L
        if 'local_z' in element:
            v_z_ref = np.array(element['local_z'])
            y_prime = np.cross(v_z_ref, x_prime)
            y_prime /= np.linalg.norm(y_prime)
            z_prime = np.cross(x_prime, y_prime)
        else:
            global_z = np.array([0.0, 0.0, 1.0])
            if np.isclose(np.abs(np.dot(x_prime, global_z)), 1.0):
                ref_vec = np.array([1.0, 0.0, 0.0])
                y_prime = np.cross(ref_vec, x_prime)
                y_prime /= np.linalg.norm(y_prime)
                z_prime = np.cross(x_prime, y_prime)
            else:
                y_prime = np.cross(global_z, x_prime)
                y_prime /= np.linalg.norm(y_prime)
                z_prime = np.cross(x_prime, y_prime)
        R = np.vstack([x_prime, y_prime, z_prime])
        Gamma = np.zeros((12, 12))
        for k in range(4):
            Gamma[3 * k:3 * (k + 1), 3 * k:3 * (k + 1)] = R
        k_global = Gamma.T @ k_local @ Gamma
        dofs = np.concatenate((np.arange(6 * i, 6 * i + 6), np.arange(6 * j, 6 * j + 6)))
        ix_ = np.ix_(dofs, dofs)
        K[ix_] += k_global
    return K
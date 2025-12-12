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
    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes
    K = np.zeros((n_dof, n_dof))
    for el in elements:
        i = el['node_i']
        j = el['node_j']
        p_i = node_coords[i]
        p_j = node_coords[j]
        diff = p_j - p_i
        L = np.linalg.norm(diff)
        if L == 0:
            continue
        x_loc = diff / L
        if 'local_z' in el and el['local_z'] is not None:
            v_ref = np.array(el['local_z'], dtype=float)
            v_ref = v_ref / np.linalg.norm(v_ref)
        elif abs(x_loc[2]) > 0.999999:
            v_ref = np.array([0.0, 1.0, 0.0])
        else:
            v_ref = np.array([0.0, 0.0, 1.0])
        tmp_y = np.cross(v_ref, x_loc)
        norm_y = np.linalg.norm(tmp_y)
        if norm_y < 1e-12:
            if abs(x_loc[2]) > 0.999999:
                v_ref_safe = np.array([0.0, 1.0, 0.0])
            else:
                v_ref_safe = np.array([0.0, 0.0, 1.0])
            tmp_y = np.cross(v_ref_safe, x_loc)
            norm_y = np.linalg.norm(tmp_y)
        y_loc = tmp_y / norm_y
        z_loc = np.cross(x_loc, y_loc)
        R = np.array([x_loc, y_loc, z_loc])
        Gamma = np.zeros((12, 12))
        Gamma[0:3, 0:3] = R
        Gamma[3:6, 3:6] = R
        Gamma[6:9, 6:9] = R
        Gamma[9:12, 9:12] = R
        E = el['E']
        nu = el['nu']
        A = el['A']
        Iy = el['I_y']
        Iz = el['I_z']
        J = el['J']
        G = E / (2.0 * (1.0 + nu))
        k_loc = np.zeros((12, 12))
        X = E * A / L
        T = G * J / L
        Yz1 = 12 * E * Iz / L ** 3
        Yz2 = 6 * E * Iz / L ** 2
        Yz3 = 4 * E * Iz / L
        Yz4 = 2 * E * Iz / L
        Yy1 = 12 * E * Iy / L ** 3
        Yy2 = 6 * E * Iy / L ** 2
        Yy3 = 4 * E * Iy / L
        Yy4 = 2 * E * Iy / L
        k_loc[0, 0] = X
        k_loc[0, 6] = -X
        k_loc[6, 0] = -X
        k_loc[6, 6] = X
        k_loc[3, 3] = T
        k_loc[3, 9] = -T
        k_loc[9, 3] = -T
        k_loc[9, 9] = T
        k_loc[1, 1] = Yz1
        k_loc[1, 5] = Yz2
        k_loc[1, 7] = -Yz1
        k_loc[1, 11] = Yz2
        k_loc[5, 1] = Yz2
        k_loc[5, 5] = Yz3
        k_loc[5, 7] = -Yz2
        k_loc[5, 11] = Yz4
        k_loc[7, 1] = -Yz1
        k_loc[7, 5] = -Yz2
        k_loc[7, 7] = Yz1
        k_loc[7, 11] = -Yz2
        k_loc[11, 1] = Yz2
        k_loc[11, 5] = Yz4
        k_loc[11, 7] = -Yz2
        k_loc[11, 11] = Yz3
        k_loc[2, 2] = Yy1
        k_loc[2, 4] = -Yy2
        k_loc[2, 8] = -Yy1
        k_loc[2, 10] = -Yy2
        k_loc[4, 2] = -Yy2
        k_loc[4, 4] = Yy3
        k_loc[4, 8] = Yy2
        k_loc[4, 10] = Yy4
        k_loc[8, 2] = -Yy1
        k_loc[8, 4] = Yy2
        k_loc[8, 8] = Yy1
        k_loc[8, 10] = Yy2
        k_loc[10, 2] = -Yy2
        k_loc[10, 4] = Yy4
        k_loc[10, 8] = Yy2
        k_loc[10, 10] = Yy3
        k_glob = Gamma.T @ k_loc @ Gamma
        idx = np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]
        K[np.ix_(idx, idx)] += k_glob
    return K
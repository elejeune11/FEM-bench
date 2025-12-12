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
    import numpy as np
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be an array of shape (n_nodes, 3)')
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes
    K = np.zeros((ndof, ndof), dtype=float)
    eps = 1e-12
    global_z_default = np.array([0.0, 0.0, 1.0], dtype=float)
    global_y_default = np.array([0.0, 1.0, 0.0], dtype=float)
    for el in elements:
        try:
            ni = int(el['node_i'])
            nj = int(el['node_j'])
            E = float(el['E'])
            nu = float(el['nu'])
            A = float(el['A'])
            Iy = float(el['I_y'])
            Iz = float(el['I_z'])
            J = float(el['J'])
        except KeyError as e:
            raise KeyError(f'Element is missing required key: {e}')
        if not (0 <= ni < n_nodes and 0 <= nj < n_nodes):
            raise IndexError('Element node indices out of range')
        if ni == nj:
            raise ValueError('Element cannot have the same start and end node')
        xi = node_coords[ni]
        xj = node_coords[nj]
        vec = xj - xi
        L = float(np.linalg.norm(vec))
        if not np.isfinite(L) or L <= eps:
            raise ValueError('Element length must be positive and finite')
        x_axis = vec / L
        z_ref = el.get('local_z', None)
        if z_ref is None:
            if abs(np.dot(x_axis, global_z_default)) < 1.0 - 1e-08:
                z_ref = global_z_default.copy()
            else:
                z_ref = global_y_default.copy()
        else:
            z_ref = np.asarray(z_ref, dtype=float).reshape(3)
            nrm = np.linalg.norm(z_ref)
            if nrm <= eps:
                raise ValueError('Provided local_z must be non-zero')
            z_ref = z_ref / nrm
        if abs(np.dot(z_ref, x_axis)) >= 1.0 - 1e-08:
            if el.get('local_z', None) is None:
                alt = global_y_default if np.linalg.norm(np.cross(global_y_default, x_axis)) > 1e-08 else np.array([1.0, 0.0, 0.0])
                z_ref = alt
                if abs(np.dot(z_ref, x_axis)) >= 1.0 - 1e-08:
                    raise ValueError('Failed to find a suitable default local_z not parallel to the beam axis')
            else:
                raise ValueError('Provided local_z is parallel to the beam axis')
        y_axis = np.cross(z_ref, x_axis)
        ny = np.linalg.norm(y_axis)
        if ny <= eps:
            raise ValueError('Cannot construct local y-axis; check local_z and node alignment')
        y_axis = y_axis / ny
        z_axis = np.cross(x_axis, y_axis)
        R_loc2glob = np.column_stack((x_axis, y_axis, z_axis))
        R_glob2loc = R_loc2glob.T
        Gamma = np.zeros((12, 12), dtype=float)
        Gamma[0:3, 0:3] = R_glob2loc
        Gamma[3:6, 3:6] = R_glob2loc
        Gamma[6:9, 6:9] = R_glob2loc
        Gamma[9:12, 9:12] = R_glob2loc
        if abs(1.0 + nu) <= eps:
            raise ValueError("Invalid Poisson's ratio leading to division by zero in shear modulus")
        G = E / (2.0 * (1.0 + nu))
        Kl = np.zeros((12, 12), dtype=float)
        k_ax = E * A / L
        Kl[0, 0] += k_ax
        Kl[0, 6] += -k_ax
        Kl[6, 0] += -k_ax
        Kl[6, 6] += k_ax
        k_t = G * J / L
        Kl[3, 3] += k_t
        Kl[3, 9] += -k_t
        Kl[9, 3] += -k_t
        Kl[9, 9] += k_t
        b1 = 12.0 * E * Iz / L ** 3
        b2 = 6.0 * E * Iz / L ** 2
        b3 = 4.0 * E * Iz / L
        b4 = 2.0 * E * Iz / L
        Kl[1, 1] += b1
        Kl[1, 5] += b2
        Kl[1, 7] += -b1
        Kl[1, 11] += b2
        Kl[5, 1] += b2
        Kl[5, 5] += b3
        Kl[5, 7] += -b2
        Kl[5, 11] += b4
        Kl[7, 1] += -b1
        Kl[7, 5] += -b2
        Kl[7, 7] += b1
        Kl[7, 11] += -b2
        Kl[11, 1] += b2
        Kl[11, 5] += b4
        Kl[11, 7] += -b2
        Kl[11, 11] += b3
        c1 = 12.0 * E * Iy / L ** 3
        c2 = 6.0 * E * Iy / L ** 2
        c3 = 4.0 * E * Iy / L
        c4 = 2.0 * E * Iy / L
        Kl[2, 2] += c1
        Kl[2, 4] += c2
        Kl[2, 8] += -c1
        Kl[2, 10] += c2
        Kl[4, 2] += c2
        Kl[4, 4] += c3
        Kl[4, 8] += -c2
        Kl[4, 10] += c4
        Kl[8, 2] += -c1
        Kl[8, 4] += -c2
        Kl[8, 8] += c1
        Kl[8, 10] += -c2
        Kl[10, 2] += c2
        Kl[10, 4] += c4
        Kl[10, 8] += -c2
        Kl[10, 10] += c3
        Kg = Gamma.T @ Kl @ Gamma
        dof_i = np.arange(6 * ni, 6 * ni + 6)
        dof_j = np.arange(6 * nj, 6 * nj + 6)
        dofs = np.concatenate((dof_i, dof_j))
        K[np.ix_(dofs, dofs)] += Kg
    return K
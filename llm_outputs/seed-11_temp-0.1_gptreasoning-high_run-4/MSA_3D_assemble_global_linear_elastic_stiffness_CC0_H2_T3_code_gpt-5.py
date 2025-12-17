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
        raise ValueError('node_coords must be an array of shape (n_nodes, 3).')
    n_nodes = node_coords.shape[0]
    dof_per_node = 6
    K = np.zeros((dof_per_node * n_nodes, dof_per_node * n_nodes), dtype=float)
    eps = 1e-12
    z_global = np.array([0.0, 0.0, 1.0])
    y_global = np.array([0.0, 1.0, 0.0])
    for e in elements:
        i = int(e['node_i'])
        j = int(e['node_j'])
        if not (0 <= i < n_nodes and 0 <= j < n_nodes):
            raise IndexError('Element node indices out of range.')
        if i == j:
            raise ValueError('Element cannot connect a node to itself.')
        xi = node_coords[i]
        xj = node_coords[j]
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if L <= eps:
            raise ValueError('Element length is zero or too small.')
        x_dir = dx / L
        z_ref = e.get('local_z', None)
        if z_ref is None:
            z_candidate = z_global.copy()
            if abs(np.dot(z_candidate, x_dir)) > 1.0 - 1e-08:
                z_candidate = y_global.copy()
            z_ref = z_candidate
        z_ref = np.asarray(z_ref, dtype=float)
        nz = np.linalg.norm(z_ref)
        if nz <= eps:
            raise ValueError('Provided local_z vector has zero magnitude.')
        z_ref = z_ref / nz
        dot_zx = abs(np.dot(z_ref, x_dir))
        if dot_zx > 1.0 - 1e-08:
            if e.get('local_z', None) is not None:
                raise ValueError('Provided local_z is parallel to the element axis.')
            z_ref = y_global.copy()
            if abs(np.dot(z_ref, x_dir)) > 1.0 - 1e-08:
                z_ref = np.array([1.0, 0.0, 0.0])
        y_dir = np.cross(z_ref, x_dir)
        ny = np.linalg.norm(y_dir)
        if ny <= eps:
            raise ValueError('Cannot construct a valid local y-axis; local_z nearly parallel to element axis.')
        y_dir = y_dir / ny
        z_dir = np.cross(x_dir, y_dir)
        nz2 = np.linalg.norm(z_dir)
        if nz2 <= eps:
            raise ValueError('Cannot construct a valid local z-axis.')
        z_dir = z_dir / nz2
        R = np.vstack((x_dir, y_dir, z_dir))
        T = np.zeros((12, 12), dtype=float)
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        E = float(e['E'])
        nu = float(e['nu'])
        A = float(e['A'])
        Iy = float(e['I_y'])
        Iz = float(e['I_z'])
        J = float(e['J'])
        G = E / (2.0 * (1.0 + nu))
        k_local = np.zeros((12, 12), dtype=float)
        EA_L = E * A / L
        GJ_L = G * J / L
        k_local[0, 0] += EA_L
        k_local[0, 6] += -EA_L
        k_local[6, 0] += -EA_L
        k_local[6, 6] += EA_L
        k_local[3, 3] += GJ_L
        k_local[3, 9] += -GJ_L
        k_local[9, 3] += -GJ_L
        k_local[9, 9] += GJ_L
        EIz = E * Iz
        az = 12.0 * EIz / L ** 3
        bz = 6.0 * EIz / L ** 2
        cz = 4.0 * EIz / L
        dz = 2.0 * EIz / L
        k_local[1, 1] += az
        k_local[1, 5] += bz
        k_local[1, 7] += -az
        k_local[1, 11] += bz
        k_local[5, 1] += bz
        k_local[5, 5] += cz
        k_local[5, 7] += -bz
        k_local[5, 11] += dz
        k_local[7, 1] += -az
        k_local[7, 5] += -bz
        k_local[7, 7] += az
        k_local[7, 11] += -bz
        k_local[11, 1] += bz
        k_local[11, 5] += dz
        k_local[11, 7] += -bz
        k_local[11, 11] += cz
        EIy = E * Iy
        ay = 12.0 * EIy / L ** 3
        by = 6.0 * EIy / L ** 2
        cy = 4.0 * EIy / L
        dy = 2.0 * EIy / L
        k_local[2, 2] += ay
        k_local[2, 4] += -by
        k_local[2, 8] += -ay
        k_local[2, 10] += -by
        k_local[4, 2] += -by
        k_local[4, 4] += cy
        k_local[4, 8] += by
        k_local[4, 10] += dy
        k_local[8, 2] += -ay
        k_local[8, 4] += by
        k_local[8, 8] += ay
        k_local[8, 10] += by
        k_local[10, 2] += -by
        k_local[10, 4] += dy
        k_local[10, 8] += by
        k_local[10, 10] += cy
        k_global_elem = T.T @ k_local @ T
        dof_i = np.arange(6 * i, 6 * i + 6)
        dof_j = np.arange(6 * j, 6 * j + 6)
        dofs = np.concatenate((dof_i, dof_j))
        K[np.ix_(dofs, dofs)] += k_global_elem
    return K
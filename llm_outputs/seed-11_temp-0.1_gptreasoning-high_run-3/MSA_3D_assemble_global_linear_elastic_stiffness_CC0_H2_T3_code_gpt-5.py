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
    n_dof = 6 * n_nodes
    K = np.zeros((n_dof, n_dof), dtype=float)
    tol = 1e-12
    global_y = np.array([0.0, 1.0, 0.0], dtype=float)
    global_z = np.array([0.0, 0.0, 1.0], dtype=float)
    for el in elements:
        try:
            i = int(el['node_i'])
            j = int(el['node_j'])
            E = float(el['E'])
            nu = float(el['nu'])
            A = float(el['A'])
            I_y = float(el['I_y'])
            I_z = float(el['I_z'])
            J = float(el['J'])
        except KeyError as e:
            raise KeyError(f'Element is missing required key: {e}') from e
        if not (0 <= i < n_nodes and 0 <= j < n_nodes):
            raise IndexError('Element node indices out of bounds')
        xi = node_coords[i]
        xj = node_coords[j]
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if L <= tol:
            raise ValueError('Element length is zero or too small')
        e_x = dx / L
        user_local_z_supplied = 'local_z' in el and el['local_z'] is not None
        if user_local_z_supplied:
            v = np.asarray(el['local_z'], dtype=float).reshape(-1)
            if v.size != 3:
                raise ValueError('local_z must be a 3-component vector')
            v_norm = np.linalg.norm(v)
            if v_norm <= tol:
                raise ValueError('local_z vector must be non-zero')
            v = v / v_norm
            proj = v - np.dot(v, e_x) * e_x
            nproj = np.linalg.norm(proj)
            if nproj <= 1e-08:
                raise ValueError('Provided local_z is parallel to the beam axis')
            e_z = proj / nproj
        else:
            if abs(np.dot(e_x, global_z)) >= 1.0 - 1e-08:
                v = global_y
            else:
                v = global_z
            proj = v - np.dot(v, e_x) * e_x
            nproj = np.linalg.norm(proj)
            if nproj <= 1e-12:
                v = global_y
                proj = v - np.dot(v, e_x) * e_x
                nproj = np.linalg.norm(proj)
                if nproj <= 1e-12:
                    tmp = np.array([1.0, 0.0, 0.0])
                    if abs(np.dot(tmp, e_x)) > 0.9:
                        tmp = np.array([0.0, 1.0, 0.0])
                    proj = tmp - np.dot(tmp, e_x) * e_x
                    nproj = np.linalg.norm(proj)
                    if nproj <= tol:
                        raise ValueError('Failed to determine a valid local_z direction')
            e_z = proj / nproj
        e_y = np.cross(e_z, e_x)
        ny = np.linalg.norm(e_y)
        if ny <= tol:
            raise ValueError('Failed to construct orthonormal local axes')
        e_y = e_y / ny
        R = np.vstack((e_x, e_y, e_z))
        G = E / (2.0 * (1.0 + nu))
        K_loc = np.zeros((12, 12), dtype=float)
        k_ax = E * A / L
        idx_ax = [0, 6]
        K_loc[np.ix_(idx_ax, idx_ax)] += np.array([[k_ax, -k_ax], [-k_ax, k_ax]], dtype=float)
        k_tor = G * J / L
        idx_tor = [3, 9]
        K_loc[np.ix_(idx_tor, idx_tor)] += np.array([[k_tor, -k_tor], [-k_tor, k_tor]], dtype=float)
        kz1 = 12.0 * E * I_z / L ** 3
        kz2 = 6.0 * E * I_z / L ** 2
        kz3 = 4.0 * E * I_z / L
        kz4 = 2.0 * E * I_z / L
        dofs_z = [1, 5, 7, 11]
        K_bz = np.array([[kz1, kz2, -kz1, kz2], [kz2, kz3, -kz2, kz4], [-kz1, -kz2, kz1, -kz2], [kz2, kz4, -kz2, kz3]], dtype=float)
        K_loc[np.ix_(dofs_z, dofs_z)] += K_bz
        ky1 = 12.0 * E * I_y / L ** 3
        ky2 = 6.0 * E * I_y / L ** 2
        ky3 = 4.0 * E * I_y / L
        ky4 = 2.0 * E * I_y / L
        dofs_y = [2, 4, 8, 10]
        K_by = np.array([[ky1, -ky2, -ky1, -ky2], [-ky2, ky3, ky2, ky4], [-ky1, ky2, ky1, ky2], [-ky2, ky4, ky2, ky3]], dtype=float)
        K_loc[np.ix_(dofs_y, dofs_y)] += K_by
        Gamma = np.zeros((12, 12), dtype=float)
        Gamma[0:3, 0:3] = R
        Gamma[3:6, 3:6] = R
        Gamma[6:9, 6:9] = R
        Gamma[9:12, 9:12] = R
        K_el_g = Gamma.T @ K_loc @ Gamma
        edofs = [6 * i + d for d in range(6)] + [6 * j + d for d in range(6)]
        K[np.ix_(edofs, edofs)] += K_el_g
    return K
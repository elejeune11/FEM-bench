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
        raise ValueError('node_coords must be of shape (n_nodes, 3)')
    n_nodes = node_coords.shape[0]
    K = np.zeros((6 * n_nodes, 6 * n_nodes), dtype=float)
    for el in elements:
        ni = el['node_i']
        nj = el['node_j']
        xi = np.asarray(node_coords[ni], dtype=float)
        xj = np.asarray(node_coords[nj], dtype=float)
        E = float(el['E'])
        nu = float(el['nu'])
        A = float(el['A'])
        I_y = float(el['I_y'])
        I_z = float(el['I_z'])
        J = float(el['J'])
        v = xj - xi
        L = np.linalg.norm(v)
        if L <= 0:
            raise ValueError('Element has zero length between nodes {} and {}'.format(ni, nj))
        x_local = v / L
        ref = el.get('local_z', None)
        if ref is None:
            global_z = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(x_local, global_z)) > 0.999999:
                ref = np.array([0.0, 1.0, 0.0], dtype=float)
            else:
                ref = global_z
        ref = np.asarray(ref, dtype=float)
        if np.linalg.norm(ref) == 0:
            raise ValueError('local_z reference vector must be non-zero')
        ref = ref / np.linalg.norm(ref)
        if abs(np.dot(ref, x_local)) > 0.999999:
            alt = np.array([0.0, 1.0, 0.0], dtype=float)
            if abs(np.dot(alt, x_local)) > 0.999999:
                alt = np.array([1.0, 0.0, 0.0], dtype=float)
            ref = alt / np.linalg.norm(alt)
        y_local = np.cross(ref, x_local)
        ynorm = np.linalg.norm(y_local)
        if ynorm == 0:
            raise ValueError('Invalid orientation: reference vector parallel to beam axis')
        y_local = y_local / ynorm
        z_local = np.cross(x_local, y_local)
        z_local = z_local / np.linalg.norm(z_local)
        R = np.column_stack((x_local, y_local, z_local))
        Kl = np.zeros((12, 12), dtype=float)
        EA_L = E * A / L
        Kl[0, 0] = EA_L
        Kl[0, 6] = -EA_L
        Kl[6, 0] = -EA_L
        Kl[6, 6] = EA_L
        G = E / (2.0 * (1.0 + nu))
        GJ_L = G * J / L
        Kl[3, 3] = GJ_L
        Kl[3, 9] = -GJ_L
        Kl[9, 3] = -GJ_L
        Kl[9, 9] = GJ_L
        k_bz = E * I_z / L ** 3
        inds_bz = [1, 5, 7, 11]
        mat_b = np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L * L, -6.0 * L, 2.0 * L * L], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L * L, -6.0 * L, 4.0 * L * L]], dtype=float)
        mat_bz = k_bz * mat_b
        for r in range(4):
            for c in range(4):
                Kl[inds_bz[r], inds_bz[c]] += mat_bz[r, c]
        k_by = E * I_y / L ** 3
        inds_by = [2, 4, 8, 10]
        mat_by = k_by * mat_b
        for r in range(4):
            for c in range(4):
                Kl[inds_by[r], inds_by[c]] += mat_by[r, c]
        block6 = np.zeros((6, 6), dtype=float)
        block6[0:3, 0:3] = R
        block6[3:6, 3:6] = R
        T = np.zeros((12, 12), dtype=float)
        T[0:6, 0:6] = block6
        T[6:12, 6:12] = block6
        Ke_global = T.dot(Kl).dot(T.T)
        dof_i = ni * 6
        dof_j = nj * 6
        K[dof_i:dof_i + 6, dof_i:dof_i + 6] += Ke_global[0:6, 0:6]
        K[dof_i:dof_i + 6, dof_j:dof_j + 6] += Ke_global[0:6, 6:12]
        K[dof_j:dof_j + 6, dof_i:dof_i + 6] += Ke_global[6:12, 0:6]
        K[dof_j:dof_j + 6, dof_j:dof_j + 6] += Ke_global[6:12, 6:12]
    return K
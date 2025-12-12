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
    coords = np.asarray(node_coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError('node_coords must be (n_nodes, 3)')
    n_nodes = coords.shape[0]
    ndof = 6 * n_nodes
    K = np.zeros((ndof, ndof), dtype=float)
    for elem in elements:
        ni = int(elem['node_i'])
        nj = int(elem['node_j'])
        if ni > n_nodes - 1:
            ni = ni - 1
        if nj > n_nodes - 1:
            nj = nj - 1
        if ni < 0 or nj < 0 or ni >= n_nodes or (nj >= n_nodes):
            raise IndexError('Element node index out of range')
        xi = coords[ni]
        xj = coords[nj]
        d = xj - xi
        L = np.linalg.norm(d)
        if L <= 0.0:
            raise ValueError('Element length is zero or negative')
        e_x = d / L
        ref = elem.get('local_z', None)
        if ref is None:
            ref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(e_x, ref)) > 0.999999:
                ref = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            ref = np.asarray(ref, dtype=float)
            if ref.shape != (3,):
                raise ValueError('local_z must be shape (3,)')
            norm_ref = np.linalg.norm(ref)
            if norm_ref < 1e-12:
                ref = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                ref = ref / norm_ref
            if abs(np.dot(e_x, ref)) > 0.999999:
                ref = np.array([0.0, 1.0, 0.0], dtype=float)
                if abs(np.dot(e_x, ref)) > 0.999999:
                    ref = np.array([1.0, 0.0, 0.0], dtype=float)
        temp = ref - np.dot(ref, e_x) * e_x
        tnorm = np.linalg.norm(temp)
        if tnorm < 1e-12:
            alt = np.array([0.0, 1.0, 0.0], dtype=float)
            temp = alt - np.dot(alt, e_x) * e_x
            tnorm = np.linalg.norm(temp)
            if tnorm < 1e-12:
                alt = np.array([1.0, 0.0, 0.0], dtype=float)
                temp = alt - np.dot(alt, e_x) * e_x
                tnorm = np.linalg.norm(temp)
                if tnorm < 1e-12:
                    raise ValueError('Cannot determine element orientation')
        e_z = temp / tnorm
        e_y = np.cross(e_z, e_x)
        e_y /= np.linalg.norm(e_y)
        e_z = np.cross(e_x, e_y)
        e_z /= np.linalg.norm(e_z)
        R = np.column_stack((e_x, e_y, e_z))
        E = float(elem['E'])
        nu = float(elem.get('nu', 0.0))
        A = float(elem['A'])
        I_y = float(elem['I_y'])
        I_z = float(elem['I_z'])
        J = float(elem['J'])
        G = E / (2.0 * (1.0 + nu))
        K_l = np.zeros((12, 12), dtype=float)
        k_axial = E * A / L
        K_l[0, 0] = k_axial
        K_l[0, 6] = -k_axial
        K_l[6, 0] = -k_axial
        K_l[6, 6] = k_axial
        k_tors = G * J / L
        K_l[3, 3] = k_tors
        K_l[3, 9] = -k_tors
        K_l[9, 3] = -k_tors
        K_l[9, 9] = k_tors
        k_z = E * I_z / L ** 3
        mat_z = k_z * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L ** 2, -6.0 * L, 2.0 * L ** 2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L ** 2, -6.0 * L, 4.0 * L ** 2]])
        idx_z = [1, 5, 7, 11]
        for a in range(4):
            for b in range(4):
                K_l[idx_z[a], idx_z[b]] += mat_z[a, b]
        k_y = E * I_y / L ** 3
        mat_y = k_y * np.array([[12.0, -6.0 * L, -12.0, -6.0 * L], [-6.0 * L, 4.0 * L ** 2, 6.0 * L, 2.0 * L ** 2], [-12.0, 6.0 * L, 12.0, -6.0 * L], [-6.0 * L, 2.0 * L ** 2, -6.0 * L, 4.0 * L ** 2]])
        idx_y = [2, 4, 8, 10]
        for a in range(4):
            for b in range(4):
                K_l[idx_y[a], idx_y[b]] += mat_y[a, b]
        T = np.zeros((12, 12), dtype=float)
        for b in range(4):
            T[3 * b:3 * b + 3, 3 * b:3 * b + 3] = R
        K_e_g = T @ K_l @ T.T
        dof_map = [6 * ni + i for i in range(6)] + [6 * nj + i for i in range(6)]
        for a in range(12):
            Ia = dof_map[a]
            for b in range(12):
                Ib = dof_map[b]
                K[Ia, Ib] += K_e_g[a, b]
    return K
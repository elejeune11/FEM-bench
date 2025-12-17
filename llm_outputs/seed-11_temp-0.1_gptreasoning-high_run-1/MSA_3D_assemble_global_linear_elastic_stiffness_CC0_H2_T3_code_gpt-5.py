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
    K_global = np.zeros((ndof, ndof), dtype=float)
    all_indices = []
    for el in elements:
        if 'node_i' not in el or 'node_j' not in el:
            raise KeyError("Each element must contain 'node_i' and 'node_j' keys.")
        all_indices.extend([int(el['node_i']), int(el['node_j'])])
    all_indices = np.array(all_indices, dtype=int)
    all_in_0 = np.all((all_indices >= 0) & (all_indices < n_nodes))
    all_in_1 = np.all((all_indices >= 1) & (all_indices <= n_nodes))
    use_offset = 0
    if not all_in_0 and all_in_1:
        use_offset = 1
    elif not all_in_0 and (not all_in_1):
        raise IndexError('Element node indices are out of bounds for provided node_coords.')
    tol = 1e-12
    global_z_default = np.array([0.0, 0.0, 1.0])
    global_y_default = np.array([0.0, 1.0, 0.0])
    for el in elements:
        i = int(el['node_i']) - use_offset
        j = int(el['node_j']) - use_offset
        if i < 0 or i >= n_nodes or j < 0 or (j >= n_nodes):
            raise IndexError('Element node index out of range after normalization.')
        xi = node_coords[i]
        xj = node_coords[j]
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if not np.isfinite(L) or L <= tol:
            raise ValueError('Element has zero or invalid length.')
        try:
            E = float(el['E'])
            nu = float(el['nu'])
            A = float(el['A'])
            Iy = float(el['I_y'])
            Iz = float(el['I_z'])
            J = float(el['J'])
        except KeyError as e:
            raise KeyError(f'Missing required element property: {e}')
        if any((param <= 0.0 for param in [E, A, Iy, Iz, J])):
            raise ValueError('Section and material properties E, A, I_y, I_z, and J must be positive.')
        G = E / (2.0 * (1.0 + nu))
        ex = dx / L
        z_ref = None
        if 'local_z' in el and el['local_z'] is not None:
            z_ref = np.asarray(el['local_z'], dtype=float)
            if z_ref.shape != (3,):
                raise ValueError('local_z must be an array-like of shape (3,)')
            norm_z = np.linalg.norm(z_ref)
            if norm_z <= tol:
                raise ValueError('Provided local_z must be a non-zero vector.')
            z_ref = z_ref / norm_z
            if abs(np.dot(z_ref, ex)) >= 1.0 - 1e-09:
                raise ValueError('Provided local_z is parallel to the beam axis, which is not allowed.')
        else:
            z_ref = global_z_default.copy()
            if abs(np.dot(z_ref, ex)) >= 1.0 - 1e-09:
                z_ref = global_y_default.copy()
        ey_temp = np.cross(z_ref, ex)
        norm_ey = np.linalg.norm(ey_temp)
        if norm_ey <= tol:
            z_ref = global_y_default.copy()
            ey_temp = np.cross(z_ref, ex)
            norm_ey = np.linalg.norm(ey_temp)
            if norm_ey <= tol:
                raise ValueError('Unable to define a valid local coordinate system (degenerate orientation).')
        ey = ey_temp / norm_ey
        ez = np.cross(ex, ey)
        ez_norm = np.linalg.norm(ez)
        if ez_norm <= tol:
            raise ValueError('Failed to construct orthonormal local axes.')
        ez = ez / ez_norm
        R = np.column_stack((ex, ey, ez))
        Rt = R.T
        Gamma = np.zeros((12, 12), dtype=float)
        Gamma[0:3, 0:3] = Rt
        Gamma[3:6, 3:6] = Rt
        Gamma[6:9, 6:9] = Rt
        Gamma[9:12, 9:12] = Rt
        K_l = np.zeros((12, 12), dtype=float)
        k_ax = E * A / L
        K_l[0, 0] += k_ax
        K_l[0, 6] -= k_ax
        K_l[6, 0] -= k_ax
        K_l[6, 6] += k_ax
        k_tor = G * J / L
        K_l[3, 3] += k_tor
        K_l[3, 9] -= k_tor
        K_l[9, 3] -= k_tor
        K_l[9, 9] += k_tor
        EIz = E * Iz
        c1z = 12.0 * EIz / L ** 3
        c2z = 6.0 * EIz / L ** 2
        c3z = 4.0 * EIz / L
        c4z = 2.0 * EIz / L
        vi, tiz, vj, tjz = (1, 5, 7, 11)
        K_l[vi, vi] += c1z
        K_l[vi, tiz] += c2z
        K_l[vi, vj] += -c1z
        K_l[vi, tjz] += c2z
        K_l[tiz, vi] += c2z
        K_l[tiz, tiz] += c3z
        K_l[tiz, vj] += -c2z
        K_l[tiz, tjz] += c4z
        K_l[vj, vi] += -c1z
        K_l[vj, tiz] += -c2z
        K_l[vj, vj] += c1z
        K_l[vj, tjz] += -c2z
        K_l[tjz, vi] += c2z
        K_l[tjz, tiz] += c4z
        K_l[tjz, vj] += -c2z
        K_l[tjz, tjz] += c3z
        EIy = E * Iy
        c1y = 12.0 * EIy / L ** 3
        c2y = 6.0 * EIy / L ** 2
        c3y = 4.0 * EIy / L
        c4y = 2.0 * EIy / L
        wi, ti_y, wj, tj_y = (2, 4, 8, 10)
        K_l[wi, wi] += c1y
        K_l[wi, ti_y] += -c2y
        K_l[wi, wj] += -c1y
        K_l[wi, tj_y] += -c2y
        K_l[ti_y, wi] += -c2y
        K_l[ti_y, ti_y] += c3y
        K_l[ti_y, wj] += c2y
        K_l[ti_y, tj_y] += c4y
        K_l[wj, wi] += -c1y
        K_l[wj, ti_y] += c2y
        K_l[wj, wj] += c1y
        K_l[wj, tj_y] += c2y
        K_l[tj_y, wi] += -c2y
        K_l[tj_y, ti_y] += c4y
        K_l[tj_y, wj] += c2y
        K_l[tj_y, tj_y] += c3y
        K_e_g = Gamma.T @ K_l @ Gamma
        dof_map = list(range(6 * i, 6 * i + 6)) + list(range(6 * j, 6 * j + 6))
        idx = np.ix_(dof_map, dof_map)
        K_global[idx] = K_global[idx] + K_e_g
    return K_global
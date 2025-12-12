def MSA_3D_assemble_global_geometric_stiffness_CC1_H4_T2(node_coords: np.ndarray, elements: Sequence[dict], u_global: np.ndarray) -> np.ndarray:
    """
    Assemble the global geometric (initial-stress) stiffness matrix K_g for a 3D frame
    under a given global displacement state.
    Each 2-node Euler–Bernoulli beam contributes a 12×12 local geometric stiffness
    matrix k_g^local that depends on the element length and the internal end
    force/moment resultants induced by the current displacement state. The local
    matrix is then mapped to global coordinates with a 12×12 direction-cosine
    transformation Γ and scattered into the global K_g.
    Parameters
    ----------
    node_coords : (n_nodes, 3) ndarray of float
        Global Cartesian coordinates [x, y, z] of each node (0-based indexing).
    elements : sequence of dict
        Per-element dictionaries. Required keys per element:
            'E' : float
                Young's modulus (Pa).
            'nu' : float
                Poisson's ratio (unitless).
            'A' : float
                Cross-sectional area (m²).
            'I_y', 'I_z' : float
                Second moments of area about the local y- and z-axes (m⁴).
            'J' : float
                Torsional constant (m⁴).
            'local_z' : array-like of shape (3,), optional
                Unit vector in global coordinates defining the local z-axis orientation.
                Must not be parallel to the beam axis. If None, a default is chosen (see Notes).
    u_global : (6*n_nodes,) ndarray of float
        Global displacement vector with 6 DOF per node in the order
        [u_x, u_y, u_z, θ_x, θ_y, θ_z] for node 0, then node 1, etc.
    Returns
    -------
    K : (6*n_nodes, 6*n_nodes) ndarray of float
        Assembled global geometric stiffness matrix. For conservative loading and
        the standard formulation, K_g is symmetric.
    """
    n_nodes = int(node_coords.shape[0])
    dof = 6 * n_nodes
    K = np.zeros((dof, dof), dtype=float)

    def _extract_element_node_indices(el):
        for key in ('nodes', 'node_indices', 'node_ids', 'connectivity', 'conn', 'nodes_idx', 'n'):
            if key in el:
                arr = np.asarray(el[key], dtype=int)
                if arr.size != 2:
                    raise ValueError('Element node index array must have length 2.')
                return (int(arr[0]), int(arr[1]))
        raise KeyError("Element dictionary missing node index key (expected one of 'nodes','node_indices','node_ids','connectivity','conn').")

    def _build_transformation_matrix(e_x, ref_vec):
        e_x = e_x / np.linalg.norm(e_x)
        v = np.asarray(ref_vec, dtype=float)
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            raise ValueError('Reference vector for local z cannot be zero.')
        v = v / v_norm
        if abs(np.dot(v, e_x)) > 1.0 - 1e-12:
            raise ValueError('Reference vector for local z is parallel to element axis.')
        e_y = np.cross(v, e_x)
        n_e_y = np.linalg.norm(e_y)
        if n_e_y < 1e-12:
            raise ValueError('Failed to build local y vector (degenerate reference).')
        e_y = e_y / n_e_y
        e_z = np.cross(e_x, e_y)
        R = np.column_stack((e_x, e_y, e_z))
        return R
    for el in elements:
        (i_node, j_node) = _extract_element_node_indices(el)
        xi = np.asarray(node_coords[i_node], dtype=float)
        xj = np.asarray(node_coords[j_node], dtype=float)
        dx = xj - xi
        L = np.linalg.norm(dx)
        if L <= 0:
            raise ValueError('Element length must be positive and non-zero.')
        e_x = dx / L
        local_z = el.get('local_z', None)
        if local_z is None:
            ref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(ref, e_x)) > 1.0 - 1e-08:
                ref = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            ref = np.asarray(local_z, dtype=float)
            ref_norm = np.linalg.norm(ref)
            if ref_norm == 0:
                raise ValueError('Provided local_z vector must be non-zero.')
            ref = ref / ref_norm
            if abs(np.dot(ref, e_x)) > 1.0 - 1e-08:
                raise ValueError('Provided local_z is parallel to element axis.')
        R = _build_transformation_matrix(e_x, ref)
        Z3 = np.zeros((3, 3), dtype=float)
        T_node = np.block([[R, Z3], [Z3, R]])
        Γ = np.zeros((12, 12), dtype=float)
        Γ[0:6, 0:6] = T_node
        Γ[6:12, 6:12] = T_node
        dof_i = slice(6 * i_node, 6 * i_node + 6)
        dof_j = slice(6 * j_node, 6 * j_node + 6)
        u_e_global = np.zeros(12, dtype=float)
        u_e_global[0:6] = u_global[dof_i]
        u_e_global[6:12] = u_global[dof_j]
        u_e_local = Γ.T.dot(u_e_global)
        E = float(el['E'])
        nu = float(el['nu'])
        A = float(el['A'])
        I_y = float(el['I_y'])
        I_z = float(el['I_z'])
        J = float(el['J'])
        G = E / (2.0 * (1.0 + nu))
        k_e = np.zeros((12, 12), dtype=float)
        k_ax = E * A / L
        k_e[0, 0] = k_ax
        k_e[0, 6] = -k_ax
        k_e[6, 0] = -k_ax
        k_e[6, 6] = k_ax
        k_t = G * J / L
        k_e[3, 3] = k_t
        k_e[3, 9] = -k_t
        k_e[9, 3] = -k_t
        k_e[9, 9] = k_t
        EIz = E * I_z
        coeff_z = EIz / L ** 3
        kz = coeff_z * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L * L, -6.0 * L, 2.0 * L * L], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L * L, -6.0 * L, 4.0 * L * L]], dtype=float)
        inds_z = [1, 5, 7, 11]
        for a in range(4):
            for b in range(4):
                k_e[inds_z[a], inds_z[b]] += kz[a, b]
        EIy = E * I_y
        coeff_y = EIy / L ** 3
        ky = coeff_y * np.array([[12.0, -6.0 * L, -12.0, -6.0 * L], [-6.0 * L, 4.0 * L * L, 6.0 * L, 2.0 * L * L], [-12.0, 6.0 * L, 12.0, 6.0 * L], [-6.0 * L, 2.0 * L * L, 6.0 * L, 4.0 * L * L]], dtype=float)
        inds_y = [2, 4, 8, 10]
        for a in range(4):
            for b in range(4):
                k_e[inds_y[a], inds_y[b]] += ky[a, b]
        f_local = k_e.dot(u_e_local)
        Fx2 = float(f_local[6])
        Mx2 = float(f_local[9])
        My1 = float(f_local[4])
        Mz1 = float(f_local[5])
        My2 = float(f_local[10])
        Mz2 = float(f_local[11])
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L=L, A=A, I_rho=J, Fx2=Fx2, Mx2=Mx2, My1=My1, Mz1=Mz1, My2=My2, Mz2=Mz2)
        k_g_global = Γ.dot(k_g_local).dot(Γ.T)
        global_dofs = np.concatenate((np.arange(6 * i_node, 6 * i_node + 6), np.arange(6 * j_node, 6 * j_node + 6)))
        for a_local in range(12):
            A_glob = global_dofs[a_local]
            for b_local in range(12):
                B_glob = global_dofs[b_local]
                K[A_glob, B_glob] += k_g_global[a_local, b_local]
    return K
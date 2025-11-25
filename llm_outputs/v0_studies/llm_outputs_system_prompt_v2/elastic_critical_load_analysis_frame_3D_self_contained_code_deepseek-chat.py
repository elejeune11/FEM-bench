def elastic_critical_load_analysis_frame_3D_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = node_coords.shape[0]
    n_dofs = 6 * n_nodes

    def assemble_global_stiffness_matrix_3D(node_coords, elements):
        K_global = np.zeros((n_dofs, n_dofs))
        for elem in elements:
            (i, j) = (elem['node_i'], elem['node_j'])
            L = np.linalg.norm(node_coords[j] - node_coords[i])
            (E, A, Iy, Iz, J, nu) = (elem['E'], elem['A'], elem['Iy'], elem['Iz'], elem['J'], elem['nu'])
            G = E / (2 * (1 + nu))
            k_local = np.zeros((12, 12))
            k_local[0, 0] = E * A / L
            k_local[6, 6] = E * A / L
            k_local[0, 6] = -E * A / L
            k_local[6, 0] = -E * A / L
            k_local[1, 1] = 12 * E * Iz / L ** 3
            k_local[7, 7] = 12 * E * Iz / L ** 3
            k_local[1, 7] = -12 * E * Iz / L ** 3
            k_local[7, 1] = -12 * E * Iz / L ** 3
            k_local[2, 2] = 12 * E * Iy / L ** 3
            k_local[8, 8] = 12 * E * Iy / L ** 3
            k_local[2, 8] = -12 * E * Iy / L ** 3
            k_local[8, 2] = -12 * E * Iy / L ** 3
            k_local[3, 3] = G * J / L
            k_local[9, 9] = G * J / L
            k_local[3, 9] = -G * J / L
            k_local[9, 3] = -G * J / L
            k_local[4, 4] = 4 * E * Iy / L
            k_local[10, 10] = 4 * E * Iy / L
            k_local[4, 10] = 2 * E * Iy / L
            k_local[10, 4] = 2 * E * Iy / L
            k_local[5, 5] = 4 * E * Iz / L
            k_local[11, 11] = 4 * E * Iz / L
            k_local[5, 11] = 2 * E * Iz / L
            k_local[11, 5] = 2 * E * Iz / L
            k_local[1, 5] = 6 * E * Iz / L ** 2
            k_local[5, 1] = 6 * E * Iz / L ** 2
            k_local[1, 11] = 6 * E * Iz / L ** 2
            k_local[11, 1] = 6 * E * Iz / L ** 2
            k_local[7, 5] = -6 * E * Iz / L ** 2
            k_local[5, 7] = -6 * E * Iz / L ** 2
            k_local[7, 11] = -6 * E * Iz / L ** 2
            k_local[11, 7] = -6 * E * Iz / L ** 2
            k_local[2, 4] = -6 * E * Iy / L ** 2
            k_local[4, 2] = -6 * E * Iy / L ** 2
            k_local[2, 10] = -6 * E * Iy / L ** 2
            k_local[10, 2] = -6 * E * Iy / L ** 2
            k_local[8, 4] = 6 * E * Iy / L ** 2
            k_local[4, 8] = 6 * E * Iy / L ** 2
            k_local[8, 10] = 6 * E * Iy / L ** 2
            k_local[10, 8] = 6 * E * Iy / L ** 2
            vec_x = node_coords[j] - node_coords[i]
            vec_x = vec_x / np.linalg.norm(vec_x)
            if elem.get('local_z') is not None:
                vec_z = np.array(elem['local_z'])
                vec_z = vec_z / np.linalg.norm(vec_z)
            elif abs(vec_x[2]) > 0.9:
                vec_z = np.array([0.0, 1.0, 0.0])
            else:
                vec_z = np.array([0.0, 0.0, 1.0])
            vec_y = np.cross(vec_z, vec_x)
            vec_y = vec_y / np.linalg.norm(vec_y)
            vec_z = np.cross(vec_x, vec_y)
            vec_z = vec_z / np.linalg.norm(vec_z)
            R = np.zeros((3, 3))
            R[:, 0] = vec_x
            R[:, 1] = vec_y
            R[:, 2] = vec_z
            T = np.zeros((12, 12))
            T[0:3, 0:3] = R
            T[3:6, 3:6] = R
            T[6:9, 6:9] = R
            T[9:12, 9:12] = R
            k_global_elem = T.T @ k_local @ T
            dof_indices = []
            for node in [i, j]:
                for dof in range(6):
                    dof_indices.append(6 * node + dof)
            for (idx_i, global_i) in enumerate(dof_indices):
                for (idx_j, global_j) in enumerate(dof_indices):
                    K_global[global_i, global_j] += k_global_elem[idx_i, idx_j]
        return K_global

    def assemble_global_load_vector_3D(nodal_loads, n_nodes):
        P_global = np.zeros(n_dofs)
        for (node_idx, load_vec) in nodal_loads.items():
            for dof in range(6):
                P_global[6 * node_idx + dof] = load_vec[dof]
        return P_global

    def apply_boundary_conditions(K, P, boundary_conditions):
        constrained_dofs = set()
        for (node_idx, bc) in boundary_conditions.items():
            if isinstance(bc[0], bool):
                for (dof_idx, is_fixed) in enumerate(bc):
                    if is_fixed:
                        constrained_dofs.add(6 * node_idx + dof_idx)
            else:
                for dof_idx in bc:
                    constrained_dofs.add(6 * node_idx + dof_idx)
        free_dofs = [i for i in range(n_dofs) if i not in constrained_dofs]
        constrained_dofs = sorted(constrained_dofs)
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        P_f = P[free_dofs]
        return (K_ff, P_f, free_dofs, constrained_dofs)

    def solve_linear_static(K_ff, P_f):
        u_f = np.linalg.solve(K_ff, P_f)
        return u_f

    def assemble_geometric_stiffness_matrix_3D(node_coords, elements, u_global):
        K_g_global = np.zeros((n_dofs, n_dofs))
        for elem in elements:
            (i, j) = (elem['node_i'], elem['node_j'])
            L = np.linalg.norm(node_coords[j] - node_coords[i])
            (A, I_rho) = (elem['A'], elem['I_rho'])
            u_elem = np.zeros(12)
            for (idx, node) in enumerate([i, j]):
                for dof in range(6):
                    u_elem[6 * idx + dof] = u_global[6 * node + dof]
            axial_force = (u_elem[6] - u_elem[0]) * elem['E'] * A / L
            k_g_local = np.zeros((12, 12))
            factor = axial_force / (30 * L)
            k_g_local[1, 1] = 36
            k_g_local[7, 7] = 36
            k_g_local[1, 7] = -36
            k_g_local[7, 1] = -36
            k_g_local[2, 2] = 36
            k_g_local[8, 8] = 36
            k_g_local[2, 8] = -36
            k_g_local[8, 2] = -36
            k_g_local[1, 5] = 3 * L
            k_g_local[5, 1] = 3 * L
            k_g_local[1, 11] = -3 * L
            k_g_local[11, 1] = -3 * L
            k_g_local[7, 5] = -3 * L
            k_g_local[5, 7] = -3 * L
            k_g_local[7, 11] = 3 * L
            k_g_local[11, 7] = 3 * L
            k_g_local[2, 4] = -3 * L
            k_g_local[4, 2] = -3 * L
            k_g_local[2, 10] = 3 * L
            k_g_local[10, 2] = 3 * L
            k_g_local[8, 4] = 3 * L
            k_g_local[4, 8] = 3 * L
            k_g_local[8, 10] = -3 * L
            k_g_local[10, 8] = -3 * L
            k_g_local[4, 4] = 4 * L ** 2
            k_g_local[10, 10] = 4 * L ** 2
            k_g_local[4, 10] = -L ** 2
            k_g_local[10, 4] = -L ** 2
            k_g_local[5, 5] = 4 * L ** 2
            k_g_local[11, 11] = 4 * L ** 2
            k_g_local[5, 11] = -L ** 2
            k_g_local[11, 5] = -L ** 2
            k_g_local[3, 3] = I_rho / A * 36 / L ** 2
            k_g_local[9, 9] = I_rho / A * 36 / L ** 2
            k_g_local[3, 9] = -I_rho / A * 36 / L ** 2
            k_g_local[9, 3] = -I_rho / A * 36 / L ** 2
            k_g_local *= factor
            vec_x = node_coords[j] - node_coords[i]
            vec_x = vec_x / np.linalg.norm(vec_x)
            if elem.get('local_z') is not None:
                vec_z = np.array(elem['local_z'])
                vec_z = vec_z / np.linalg.norm(vec_z)
            elif abs(vec_x[2]) > 0.9:
                vec_z = np.array([0.0, 1.0, 0.0])
            else:
                vec_z = np.array([0.0, 0.0, 1.0])
            vec_y = np.cross(vec_z, vec_x)
            vec_y = vec_y / np.linalg.norm(vec_y)
            vec_z = np.cross(vec_x, vec_y)
            vec_z = vec_z / np.linalg.norm(vec_z)
            R = np.zeros((3, 3))
            R[:, 0] = vec_x
            R[:, 1] = vec_y
            R[:, 2] = vec_z
            T = np.zeros((12, 12))
            T[0:3, 0:3] = R
            T[3:6, 3:6] = R
            T[6:9, 6:9] = R
            T[9:12, 9:12] = R
            k_g_global_elem = T.T @ k_g_local @ T
            dof_indices = []
            for node in [i, j]:
                for dof in range(6):
                    dof_indices.append(6 * node + dof)
            for (idx_i, global_i) in enumerate(dof_indices):
                for (idx_j, global_j) in enumerate(dof_indices):
                    K_g_global[global_i, global_j] += k_g_global_elem[idx_i, idx_j]
        return K_g_global

    def solve_buckling_eigenproblem(K_ff, K_g_ff):
        try:
            (eigenvalues, eigenvectors) = scipy.linalg.eigh(K_ff, -K_g_ff)
        except:
            (eigenvalues, eigenvectors) = scipy.linalg.eig(K_ff, -K_g_ff)
            eigenvalues = eigenvalues.real
            eigenvectors = eigenvectors.real
        positive_eigenvalues = eigenvalues[eigenvalues > 0]
        if len(positive_eigenvalues) == 0:
            raise ValueError('No positive eigenvalues found')
        min_positive_idx = np.argmin(positive_eigenvalues)
        critical_load_factor = positive_eigenvalues[min_positive_idx]
        corresponding_eigenvector = eigenvectors[:, eigenvalues == critical_load_factor]
        if corresponding_eigenvector.shape[1] > 1:
            corresponding_eigenvector = corresponding_eigenvector[:, 0]
        else:
            corresponding_eigenvector = corresponding_eigenvector.flatten()
        return (critical_load_factor, corresponding_eigenvector)
    K_global = assemble_global_stiffness_matrix_3D(node_coords, elements)
    P_global = assemble_global_load_vector_3D(nodal_loads, n_nodes)
    (K_ff, P_f, free_dofs, constrained_dofs) = apply_boundary_conditions(K_global, P_global, boundary_conditions)
    u_f = solve_linear_static(K_ff, P_f)
    u_global = np.zeros(n_dofs)
    u_global[free_dofs] = u_f
    K_g_global = assemble_geometric_stiffness_matrix_3D(node_coords, elements, u_global)
    K_g_ff = K_g_global[np.ix_(free_dofs, free_dofs)]
    (critical_load_factor, buckling_mode_f) = solve_buckling_eigenproblem(K_ff, K_g_ff)
    buckling_mode_global = np.zeros(n_dofs)
    buckling_mode_global[free_dofs] = buckling_mode_f
    return (critical_load_factor, buckling_mode_global)
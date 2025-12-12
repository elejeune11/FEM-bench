def MSA_3D_assemble_global_linear_elastic_stiffness_CC0_H2_T3(node_coords, elements):
    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes
    K = np.zeros((n_dof, n_dof))
    for elem in elements:
        (i, j) = (elem['node_i'], elem['node_j'])
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        Iy = elem['I_y']
        Iz = elem['I_z']
        J = elem['J']
        local_z = elem.get('local_z', None)
        (xi, yi, zi) = node_coords[i]
        (xj, yj, zj) = node_coords[j]
        L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
        ex = np.array([xj - xi, yj - yi, zj - zi]) / L
        if local_z is None:
            if abs(ex[2]) < 0.9:
                local_z = np.array([0, 0, 1])
            else:
                local_z = np.array([0, 1, 0])
        else:
            local_z = np.array(local_z)
            local_z /= np.linalg.norm(local_z)
        if abs(np.dot(ex, local_z)) > 0.9999:
            raise ValueError('local_z must not be parallel to the beam axis')
        ey = np.cross(ex, local_z)
        ey /= np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        Gamma = np.zeros((12, 12))
        Gamma[0, 0] = ex[0]
        Gamma[0, 1] = ex[1]
        Gamma[0, 2] = ex[2]
        Gamma[1, 0] = ey[0]
        Gamma[1, 1] = ey[1]
        Gamma[1, 2] = ey[2]
        Gamma[2, 0] = ez[0]
        Gamma[2, 1] = ez[1]
        Gamma[2, 2] = ez[2]
        Gamma[3, 3] = ex[0]
        Gamma[3, 4] = ex[1]
        Gamma[3, 5] = ex[2]
        Gamma[4, 3] = ey[0]
        Gamma[4, 4] = ey[1]
        Gamma[4, 5] = ey[2]
        Gamma[5, 3] = ez[0]
        Gamma[5, 4] = ez[1]
        Gamma[5, 5] = ez[2]
        Gamma[6, 6] = ex[0]
        Gamma[6, 7] = ex[1]
        Gamma[6, 8] = ex[2]
        Gamma[7, 6] = ey[0]
        Gamma[7, 7] = ey[1]
        Gamma[7, 8] = ey[2]
        Gamma[8, 6] = ez[0]
        Gamma[8, 7] = ez[1]
        Gamma[8, 8] = ez[2]
        Gamma[9, 9] = ex[0]
        Gamma[9, 10] = ex[1]
        Gamma[9, 11] = ex[2]
        Gamma[10, 9] = ey[0]
        Gamma[10, 10] = ey[1]
        Gamma[10, 11] = ey[2]
        Gamma[11, 9] = ez[0]
        Gamma[11, 10] = ez[1]
        Gamma[11, 11] = ez[2]
        K_local = np.zeros((12, 12))
        k_axial = E * A / L
        K_local[0, 0] = k_axial
        K_local[6, 6] = k_axial
        K_local[0, 6] = -k_axial
        K_local[6, 0] = -k_axial
        k_torsion = G = E / (2 * (1 + nu))
        k_torsion = G * J / L
        K_local[3, 3] = k_torsion
        K_local[9, 9] = k_torsion
        K_local[3, 9] = -k_torsion
        K_local[9, 3] = -k_torsion
        k_bend_y = 12 * E * Iz / L ** 3
        k_bend_y_rot = 6 * E * Iz / L ** 2
        k_bend_y_rot2 = 4 * E * Iz / L
        k_bend_y_rot3 = 2 * E * Iz / L
        K_local[1, 1] = k_bend_y
        K_local[5, 5] = k_bend_y_rot2
        K_local[1, 5] = k_bend_y_rot
        K_local[5, 1] = k_bend_y_rot
        K_local[7, 7] = k_bend_y
        K_local[11, 11] = k_bend_y_rot2
        K_local[7, 11] = -k_bend_y_rot
        K_local[11, 7] = -k_bend_y_rot
        K_local[1, 7] = -k_bend_y
        K_local[7, 1] = -k_bend_y
        K_local[5, 11] = k_bend_y_rot3
        K_local[11, 5] = k_bend_y_rot3
        k_bend_z = 12 * E * Iy / L ** 3
        k_bend_z_rot = 6 * E * Iy / L ** 2
        k_bend_z_rot2 = 4 * E * Iy / L
        k_bend_z_rot3 = 2 * E * Iy / L
        K_local[2, 2] = k_bend_z
        K_local[4, 4] = k_bend_z_rot2
        K_local[2, 4] = -k_bend_z_rot
        K_local[4, 2] = -k_bend_z_rot
        K_local[8, 8] = k_bend_z
        K_local[10, 10] = k_bend_z_rot2
        K_local[8, 10] = k_bend_z_rot
        K_local[10, 8] = k_bend_z_rot
        K_local[2, 8] = -k_bend_z
        K_local[8, 2] = -k_bend_z
        K_local[4, 10] = k_bend_z_rot3
        K_local[10, 4] = k_bend_z_rot3
        K_global = Gamma.T @ K_local @ Gamma
        dof_indices = np.array([6 * i, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5, 6 * j, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5])
        for ii in range(12):
            for jj in range(12):
                K[dof_indices[ii], dof_indices[jj]] += K_global[ii, jj]
    return K
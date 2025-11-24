def solve_linear_elastic_frame_3D_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    N = node_coords.shape[0]
    ndof = 6 * N
    K_global = np.zeros((ndof, ndof))
    P_global = np.zeros(ndof)
    for elem in elements:
        (i, j) = (elem['node_i'], elem['node_j'])
        (E, nu, A, I_y, I_z, J) = (elem['E'], elem['nu'], elem['A'], elem['I_y'], elem['I_z'], elem['J'])
        (xi, yi, zi) = node_coords[i]
        (xj, yj, zj) = node_coords[j]
        (dx, dy, dz) = (xj - xi, yj - yi, zj - zi)
        L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        G = E / (2 * (1 + nu))
        k_local = np.zeros((12, 12))
        EA_L = E * A / L
        GJ_L = G * J / L
        EIy_L = E * I_y / L
        EIy_L2 = EIy_L / L
        EIy_L3 = EIy_L2 / L
        EIz_L = E * I_z / L
        EIz_L2 = EIz_L / L
        EIz_L3 = EIz_L2 / L
        k_local[0, 0] = EA_L
        k_local[0, 6] = -EA_L
        k_local[6, 0] = -EA_L
        k_local[6, 6] = EA_L
        k_local[1, 1] = 12 * EIz_L3
        k_local[1, 5] = 6 * EIz_L2
        k_local[1, 7] = -12 * EIz_L3
        k_local[1, 11] = 6 * EIz_L2
        k_local[5, 1] = 6 * EIz_L2
        k_local[5, 5] = 4 * EIz_L
        k_local[5, 7] = -6 * EIz_L2
        k_local[5, 11] = 2 * EIz_L
        k_local[7, 1] = -12 * EIz_L3
        k_local[7, 5] = -6 * EIz_L2
        k_local[7, 7] = 12 * EIz_L3
        k_local[7, 11] = -6 * EIz_L2
        k_local[11, 1] = 6 * EIz_L2
        k_local[11, 5] = 2 * EIz_L
        k_local[11, 7] = -6 * EIz_L2
        k_local[11, 11] = 4 * EIz_L
        k_local[2, 2] = 12 * EIy_L3
        k_local[2, 4] = -6 * EIy_L2
        k_local[2, 8] = -12 * EIy_L3
        k_local[2, 10] = -6 * EIy_L2
        k_local[4, 2] = -6 * EIy_L2
        k_local[4, 4] = 4 * EIy_L
        k_local[4, 8] = 6 * EIy_L2
        k_local[4, 10] = 2 * EIy_L
        k_local[8, 2] = -12 * EIy_L3
        k_local[8, 4] = 6 * EIy_L2
        k_local[8, 8] = 12 * EIy_L3
        k_local[8, 10] = 6 * EIy_L2
        k_local[10, 2] = -6 * EIy_L2
        k_local[10, 4] = 2 * EIy_L
        k_local[10, 8] = 6 * EIy_L2
        k_local[10, 10] = 4 * EIy_L
        k_local[3, 3] = GJ_L
        k_local[3, 9] = -GJ_L
        k_local[9, 3] = -GJ_L
        k_local[9, 9] = GJ_L
        e_x = np.array([dx, dy, dz]) / L
        if elem.get('local_z') is not None:
            local_z = np.array(elem['local_z'])
            e_z = local_z - np.dot(local_z, e_x) * e_x
            e_z_norm = np.linalg.norm(e_z)
            if e_z_norm < 1e-12:
                raise ValueError('local_z is parallel to beam axis')
            e_z /= e_z_norm
        else:
            if abs(e_x[2]) > 0.9:
                e_z = np.array([0.0, 1.0, 0.0])
            else:
                e_z = np.array([0.0, 0.0, 1.0])
            e_z = e_z - np.dot(e_z, e_x) * e_x
            e_z /= np.linalg.norm(e_z)
        e_y = np.cross(e_z, e_x)
        e_y /= np.linalg.norm(e_y)
        e_z = np.cross(e_x, e_y)
        R = np.column_stack([e_x, e_y, e_z])
        T = np.zeros((12, 12))
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        k_global_elem = T @ k_local @ T.T
        dofs_i = slice(6 * i, 6 * i + 6)
        dofs_j = slice(6 * j, 6 * j + 6)
        K_global[dofs_i, dofs_i] += k_global_elem[0:6, 0:6]
        K_global[dofs_i, dofs_j] += k_global_elem[0:6, 6:12]
        K_global[dofs_j, dofs_i] += k_global_elem[6:12, 0:6]
        K_global[dofs_j, dofs_j] += k_global_elem[6:12, 6:12]
    for (node, loads) in nodal_loads.items():
        start_dof = 6 * node
        P_global[start_dof:start_dof + 6] = loads
    fixed_dofs = np.zeros(ndof, dtype=bool)
    for (node, bc) in boundary_conditions.items():
        start_dof = 6 * node
        fixed_dofs[start_dof:start_dof + 6] = bc
    free_dofs = ~fixed_dofs
    K_ff = K_global[free_dofs, :][:, free_dofs]
    K_fs = K_global[free_dofs, :][:, fixed_dofs]
    P_f = P_global[free_dofs]
    cond_num = np.linalg.cond(K_ff)
    if cond_num >= 1e+16:
        raise ValueError(f'System is ill-conditioned (cond(K_ff) = {cond_num})')
    u_f = np.linalg.solve(K_ff, P_f)
    u = np.zeros(ndof)
    u[free_dofs] = u_f
    P_s = P_global[fixed_dofs]
    u_s = u[fixed_dofs]
    r_s = K_global[fixed_dofs, :][:, free_dofs] @ u_f - P_s
    r = np.zeros(ndof)
    r[fixed_dofs] = r_s
    return (u, r)
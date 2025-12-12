def MSA_3D_local_element_loads_CC0_H2_T1(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global):
    E = ele_info['E']
    nu = ele_info['nu']
    A = ele_info['A']
    Iy = ele_info['I_y']
    Iz = ele_info['I_z']
    J = ele_info['J']
    ref_vec = ele_info.get('local_z', None)
    L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
    K_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
    Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ref_vec)
    u_dofs_local = Gamma @ np.asarray(u_dofs_global)
    load_dofs_local = K_local @ u_dofs_local
    return load_dofs_local
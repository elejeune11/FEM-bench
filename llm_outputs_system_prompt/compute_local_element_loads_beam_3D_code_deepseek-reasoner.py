def compute_local_element_loads_beam_3D(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global):
    E = ele_info['E']
    nu = ele_info['nu']
    A = ele_info['A']
    I_y = ele_info['I_y']
    I_z = ele_info['I_z']
    J = ele_info['J']
    local_z = ele_info.get('local_z')
    L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
    Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z)
    u_dofs_local = Gamma @ np.array(u_dofs_global)
    K_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, I_y, I_z, J)
    load_dofs_local = K_local @ u_dofs_local
    return load_dofs_local
def compute_local_element_loads_beam_3D(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global):
    L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
    if np.isclose(L, 0.0):
        raise ValueError('Beam length is zero.')
    ref_vec = ele_info.get('local_z', None)
    Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ref_vec)
    u_dofs_local = Gamma @ u_dofs_global
    E = ele_info['E']
    nu = ele_info['nu']
    A = ele_info['A']
    I_y = ele_info['I_y']
    I_z = ele_info['I_z']
    J = ele_info['J']
    k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, I_y, I_z, J)
    load_dofs_local = k_local @ u_dofs_local
    return load_dofs_local
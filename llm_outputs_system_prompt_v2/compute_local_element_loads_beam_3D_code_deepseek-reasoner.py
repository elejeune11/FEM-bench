def compute_local_element_loads_beam_3D(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global):
    L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
    if np.isclose(L, 0.0):
        raise ValueError('Beam length is zero.')
    ref_vec = ele_info.get('local_z', None)
    Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ref_vec)
    u_dofs_local = Gamma @ np.asarray(u_dofs_global)
    K_local = local_elastic_stiffness_matrix_3D_beam(ele_info['E'], ele_info['nu'], ele_info['A'], L, ele_info['I_y'], ele_info['I_z'], ele_info['J'])
    return K_local @ u_dofs_local
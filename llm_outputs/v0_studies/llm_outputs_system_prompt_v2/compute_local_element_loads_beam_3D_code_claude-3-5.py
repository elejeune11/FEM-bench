def compute_local_element_loads_beam_3D(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global):
    Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ele_info.get('local_z', None))
    (dx, dy, dz) = (xj - xi, yj - yi, zj - zi)
    L = np.sqrt(dx * dx + dy * dy + dz * dz)
    K_local = local_elastic_stiffness_matrix_3D_beam(ele_info['E'], ele_info['nu'], ele_info['A'], L, ele_info['I_y'], ele_info['I_z'], ele_info['J'])
    u_dofs_local = Gamma @ u_dofs_global
    load_dofs_local = K_local @ u_dofs_local
    return load_dofs_local
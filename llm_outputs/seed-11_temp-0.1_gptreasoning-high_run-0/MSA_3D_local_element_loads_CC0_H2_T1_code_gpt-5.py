def MSA_3D_local_element_loads_CC0_H2_T1(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global):
    import numpy as np
    E = ele_info['E']
    nu = ele_info['nu']
    A = ele_info['A']
    J = ele_info['J']
    Iy = ele_info['I_y'] if 'I_y' in ele_info else ele_info.get('Iy', None)
    Iz = ele_info['I_z'] if 'I_z' in ele_info else ele_info.get('Iz', None)
    if Iy is None or Iz is None:
        raise ValueError("Element second moments of area 'I_y' and 'I_z' must be provided.")
    ref_vec = ele_info.get('local_z', None)
    (dx, dy, dz) = (xj - xi, yj - yi, zj - zi)
    L = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    if np.isclose(L, 0.0):
        raise ValueError('Beam length is zero.')
    u = np.asarray(u_dofs_global, dtype=float).reshape(-1)
    if u.size != 12:
        raise ValueError('u_dofs_global must have length 12.')
    Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ref_vec)
    u_local = Gamma @ u
    k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
    load_local = k_local @ u_local
    return load_local
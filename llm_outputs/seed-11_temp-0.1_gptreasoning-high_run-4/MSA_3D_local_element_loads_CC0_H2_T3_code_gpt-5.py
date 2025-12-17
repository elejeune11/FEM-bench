def MSA_3D_local_element_loads_CC0_H2_T3(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global):
    import numpy as np
    from typing import Callable, Optional
    import pytest
    try:
        E = float(ele_info['E'])
        nu = float(ele_info['nu'])
        A = float(ele_info['A'])
        I_y = float(ele_info['I_y'])
        I_z = float(ele_info['I_z'])
        J = float(ele_info['J'])
    except KeyError as e:
        raise KeyError(f'Missing required key in ele_info: {e}')
    dx = float(xj) - float(xi)
    dy = float(yj) - float(yi)
    dz = float(zj) - float(zi)
    L = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    tol = 1e-12
    if not np.isfinite(L) or L <= tol:
        raise ValueError('Invalid element length: zero or undefined.')
    ex = np.array([dx, dy, dz], dtype=float) / L
    provided_z = 'local_z' in ele_info and ele_info['local_z'] is not None
    if provided_z:
        local_z_vec = np.asarray(ele_info['local_z'], dtype=float).reshape(-1)
        if local_z_vec.size != 3:
            raise ValueError("Invalid 'local_z': must be an array-like of length 3.")
        norm_lz = np.linalg.norm(local_z_vec)
        if not np.isfinite(norm_lz) or norm_lz <= tol:
            raise ValueError("Invalid 'local_z': zero or non-finite vector.")
        z_ref = local_z_vec / norm_lz
        cross_mag = np.linalg.norm(np.cross(z_ref, ex))
        if cross_mag <= 1e-08:
            raise ValueError("Invalid 'local_z': cannot be parallel to the element axis.")
    else:
        z_ref = np.array([0.0, 0.0, 1.0], dtype=float)
        if np.isclose(abs(np.dot(ex, z_ref)), 1.0, atol=1e-08, rtol=0.0):
            z_ref = np.array([0.0, 1.0, 0.0], dtype=float)
    ey_temp = np.cross(z_ref, ex)
    norm_ey = np.linalg.norm(ey_temp)
    if norm_ey <= 1e-12:
        axes = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
        best_axis = None
        best_cross = 0.0
        for a in axes:
            c = np.linalg.norm(np.cross(a, ex))
            if c > best_cross:
                best_cross = c
                best_axis = a
        if best_axis is None or best_cross <= 1e-12:
            raise ValueError('Failed to construct a valid local coordinate system.')
        ey_temp = np.cross(best_axis, ex)
        norm_ey = np.linalg.norm(ey_temp)
        if norm_ey <= 1e-12:
            raise ValueError('Failed to construct a valid local coordinate system.')
    ey = ey_temp / norm_ey
    ez = np.cross(ex, ey)
    ez_norm = np.linalg.norm(ez)
    if ez_norm <= 1e-12:
        raise ValueError('Failed to construct a valid local coordinate system.')
    ez = ez / ez_norm
    Q = np.column_stack((ex, ey, ez))
    T = np.zeros((12, 12), dtype=float)
    T[0:3, 0:3] = Q
    T[3:6, 3:6] = Q
    T[6:9, 6:9] = Q
    T[9:12, 9:12] = Q
    u_global = np.asarray(u_dofs_global, dtype=float).reshape(-1)
    if u_global.size != 12:
        raise ValueError('u_dofs_global must have length 12.')
    u_local = T.T @ u_global
    K = np.zeros((12, 12), dtype=float)
    EA_over_L = E * A / L
    K[0, 0] += EA_over_L
    K[0, 6] -= EA_over_L
    K[6, 0] -= EA_over_L
    K[6, 6] += EA_over_L
    G = E / (2.0 * (1.0 + nu))
    GJ_over_L = G * J / L
    K[3, 3] += GJ_over_L
    K[3, 9] -= GJ_over_L
    K[9, 3] -= GJ_over_L
    K[9, 9] += GJ_over_L
    EIz = E * I_z
    c1 = 12.0 * EIz / L ** 3
    c2 = 6.0 * EIz / L ** 2
    c3 = 4.0 * EIz / L
    c4 = 2.0 * EIz / L
    K[1, 1] += c1
    K[1, 5] += c2
    K[1, 7] -= c1
    K[1, 11] += c2
    K[5, 1] += c2
    K[5, 5] += c3
    K[5, 7] -= c2
    K[5, 11] += c4
    K[7, 1] -= c1
    K[7, 5] -= c2
    K[7, 7] += c1
    K[7, 11] -= c2
    K[11, 1] += c2
    K[11, 5] += c4
    K[11, 7] -= c2
    K[11, 11] += c3
    EIy = E * I_y
    d1 = 12.0 * EIy / L ** 3
    d2 = 6.0 * EIy / L ** 2
    d3 = 4.0 * EIy / L
    d4 = 2.0 * EIy / L
    K[2, 2] += d1
    K[2, 4] -= d2
    K[2, 8] -= d1
    K[2, 10] -= d2
    K[4, 2] -= d2
    K[4, 4] += d3
    K[4, 8] += d2
    K[4, 10] += d4
    K[8, 2] -= d1
    K[8, 4] += d2
    K[8, 8] += d1
    K[8, 10] += d2
    K[10, 2] -= d2
    K[10, 4] += d4
    K[10, 8] += d2
    K[10, 10] += d3
    load_dofs_local = K @ u_local
    return load_dofs_local.reshape(12)
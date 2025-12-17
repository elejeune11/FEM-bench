def MSA_3D_local_element_loads_CC0_H2_T3(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global):
    import numpy as np
    E = float(ele_info.get('E'))
    nu = float(ele_info.get('nu'))
    A = float(ele_info.get('A'))
    Iy = float(ele_info.get('I_y'))
    Iz = float(ele_info.get('I_z'))
    J = float(ele_info.get('J'))
    pi = np.array([xi, yi, zi], dtype=float)
    pj = np.array([xj, yj, zj], dtype=float)
    v = pj - pi
    L = float(np.linalg.norm(v))
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError('Beam length must be positive and nonzero.')
    ex = v / L
    local_z = ele_info.get('local_z', None)
    if local_z is None:
        gz = np.array([0.0, 0.0, 1.0])
        gy = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(ex, gz)) >= 1.0 - 1e-12:
            local_z = gy
        else:
            local_z = gz
    local_z = np.asarray(local_z, dtype=float).reshape(-1)
    if local_z.shape[0] != 3 or not np.all(np.isfinite(local_z)):
        raise ValueError('Invalid local_z: must be a finite 3-vector.')
    if np.linalg.norm(local_z) == 0.0:
        raise ValueError('Invalid local_z: zero vector provided.')
    if np.linalg.norm(np.cross(local_z / np.linalg.norm(local_z), ex)) <= 1e-12:
        raise ValueError('Invalid local_z: vector is parallel to the element axis.')
    ey_raw = np.cross(local_z, ex)
    norm_ey = np.linalg.norm(ey_raw)
    if norm_ey <= 0.0:
        raise ValueError('Invalid local axes: cannot construct a perpendicular local y-axis.')
    ey = ey_raw / norm_ey
    ez = np.cross(ex, ey)
    R = np.vstack((ex, ey, ez))
    T = np.zeros((12, 12), dtype=float)
    for k in range(4):
        T[3 * k:3 * (k + 1), 3 * k:3 * (k + 1)] = R
    u = np.asarray(u_dofs_global, dtype=float).reshape(-1)
    if u.size != 12:
        raise ValueError('u_dofs_global must be a 12-component vector.')
    u_local = T @ u
    G = E / (2.0 * (1.0 + nu))
    a = E * A / L
    jconst = G * J / L
    b = 12.0 * E * Iz / L ** 3
    c = 6.0 * E * Iz / L ** 2
    d = 4.0 * E * Iz / L
    econst = 2.0 * E * Iz / L
    f = 12.0 * E * Iy / L ** 3
    g = 6.0 * E * Iy / L ** 2
    h = 4.0 * E * Iy / L
    iconst = 2.0 * E * Iy / L
    k = np.zeros((12, 12), dtype=float)
    k[0, 0] = a
    k[0, 6] = -a
    k[6, 0] = -a
    k[6, 6] = a
    k[3, 3] = jconst
    k[3, 9] = -jconst
    k[9, 3] = -jconst
    k[9, 9] = jconst
    k[1, 1] = b
    k[1, 5] = c
    k[1, 7] = -b
    k[1, 11] = c
    k[5, 1] = c
    k[5, 5] = d
    k[5, 7] = -c
    k[5, 11] = econst
    k[7, 1] = -b
    k[7, 5] = -c
    k[7, 7] = b
    k[7, 11] = -c
    k[11, 1] = c
    k[11, 5] = econst
    k[11, 7] = -c
    k[11, 11] = d
    k[2, 2] = f
    k[2, 4] = -g
    k[2, 8] = -f
    k[2, 10] = -g
    k[4, 2] = -g
    k[4, 4] = h
    k[4, 8] = g
    k[4, 10] = iconst
    k[8, 2] = -f
    k[8, 4] = g
    k[8, 8] = f
    k[8, 10] = g
    k[10, 2] = -g
    k[10, 4] = iconst
    k[10, 8] = g
    k[10, 10] = h
    load_dofs_local = k @ u_local
    return load_dofs_local
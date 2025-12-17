def MSA_3D_local_element_loads_CC0_H2_T3(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global):
    import numpy as np
    u = np.asarray(u_dofs_global, dtype=float).reshape(-1)
    if u.size != 12:
        raise ValueError('u_dofs_global must be an array-like of length 12.')
    try:
        E = float(ele_info['E'])
        nu = float(ele_info['nu'])
        A = float(ele_info['A'])
        Iy = float(ele_info['I_y'])
        Iz = float(ele_info['I_z'])
        J = float(ele_info['J'])
    except KeyError as e:
        raise ValueError(f'Missing required property in ele_info: {e}')
    xi = float(xi)
    yi = float(yi)
    zi = float(zi)
    xj = float(xj)
    yj = float(yj)
    zj = float(zj)
    dx = xj - xi
    dy = yj - yi
    dz = zj - zi
    L = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    if L <= 0.0:
        raise ValueError('Beam element length must be positive and non-zero.')
    ex = np.array([dx, dy, dz], dtype=float) / L
    tol = 1e-12
    z_hint = ele_info.get('local_z', None)
    if z_hint is None:
        global_z = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(np.dot(ex, global_z)) >= 1.0 - 1e-08:
            z_dir = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            z_dir = global_z
    else:
        z_dir = np.asarray(z_hint, dtype=float).reshape(-1)
        if z_dir.size != 3:
            raise ValueError('local_z must be a 3-component vector.')
        if np.linalg.norm(z_dir) <= tol:
            raise ValueError('local_z vector must be non-zero.')
    z_proj = z_dir - np.dot(z_dir, ex) * ex
    nz = np.linalg.norm(z_proj)
    if nz <= 1e-08:
        if z_hint is None:
            raise ValueError('Default local_z selection failed due to degeneracy.')
        else:
            raise ValueError('Provided local_z must not be parallel to the beam axis.')
    ez = z_proj / nz
    ey = np.cross(ez, ex)
    ny = np.linalg.norm(ey)
    if ny <= tol:
        raise ValueError('Failed to construct a valid local coordinate system.')
    ey /= ny
    ez = np.cross(ex, ey)
    R = np.column_stack((ex, ey, ez))
    T = np.zeros((12, 12), dtype=float)
    Rt = R.T
    for k in (0, 3, 6, 9):
        T[k:k + 3, k:k + 3] = Rt
    u_local = T @ u
    G = E / (2.0 * (1.0 + nu))
    a = E * A / L
    jt = G * J / L
    b = 12.0 * E * Iz / L ** 3
    c = 6.0 * E * Iz / L ** 2
    d = 4.0 * E * Iz / L
    e2 = 2.0 * E * Iz / L
    f = 12.0 * E * Iy / L ** 3
    g = 6.0 * E * Iy / L ** 2
    h = 4.0 * E * Iy / L
    i2 = 2.0 * E * Iy / L
    K = np.zeros((12, 12), dtype=float)
    K[0, 0] = a
    K[0, 6] = -a
    K[6, 0] = -a
    K[6, 6] = a
    K[3, 3] = jt
    K[3, 9] = -jt
    K[9, 3] = -jt
    K[9, 9] = jt
    K[1, 1] = b
    K[1, 5] = c
    K[1, 7] = -b
    K[1, 11] = c
    K[5, 1] = c
    K[5, 5] = d
    K[5, 7] = -c
    K[5, 11] = e2
    K[7, 1] = -b
    K[7, 5] = -c
    K[7, 7] = b
    K[7, 11] = -c
    K[11, 1] = c
    K[11, 5] = e2
    K[11, 7] = -c
    K[11, 11] = d
    K[2, 2] = f
    K[2, 4] = -g
    K[2, 8] = -f
    K[2, 10] = -g
    K[4, 2] = -g
    K[4, 4] = h
    K[4, 8] = g
    K[4, 10] = i2
    K[8, 2] = -f
    K[8, 4] = g
    K[8, 8] = f
    K[8, 10] = g
    K[10, 2] = -g
    K[10, 4] = i2
    K[10, 8] = g
    K[10, 10] = h
    load_dofs_local = K @ u_local
    return load_dofs_local.reshape(12)
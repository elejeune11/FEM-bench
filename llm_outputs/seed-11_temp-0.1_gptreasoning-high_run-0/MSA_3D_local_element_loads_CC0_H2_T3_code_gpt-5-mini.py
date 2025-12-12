def MSA_3D_local_element_loads_CC0_H2_T3(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global):
    """
    Compute the local internal nodal force/moment vector (end loading) for a 3D Euler-Bernoulli beam element.
    applies the local stiffness matrix, and returns the corresponding internal end forces
    in the local coordinate system.
    Parameters
    ----------
    ele_info : dict
        Dictionary containing the element's material and geometric properties:
            'E' : float
                Young's modulus (Pa).
            'nu' : float
                Poisson's ratio (unitless).
            'A' : float
                Cross-sectional area (m²).
            'I_y', 'I_z' : float
                Second moments of area about the local y- and z-axes (m⁴).
            'J' : float
                Torsional constant (m⁴).
            'local_z' : array-like of shape (3,), optional
                Unit vector in global coordinates defining the local z-axis orientation.
                Must not be parallel to the beam axis. If None, a default is chosen.
                Default local_z = global z, unless the beam lies along global z — then default local_z = global y.
    xi, yi, zi : float
        Global coordinates of the element's start node.
    xj, yj, zj : float
        Global coordinates of the element's end node.
    u_dofs_global : array-like of shape (12,)
        Element displacement vector in global coordinates:
        [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2],
        where:
    Returns
    -------
    load_dofs_local : ndarray of shape (12,)
        Internal element end forces in local coordinates, ordered consistently with `u_dofs_global`.
        Positive forces/moments follow the local right-handed coordinate system conventions.
    Raises
    ------
    ValueError
        If the beam length is zero or if `local_z` is invalid.
    Notes
    -----
    elastic response to the provided displacement state, not externally applied loads.
    + local x-axis → element axis from node i to node j
    + local y-axis → perpendicular to both local z (provided) and x
    + local z-axis → defined by the provided (optional) or default orientation vector.
    [Fx_i, Fy_i, Fz_i, Mx_i, My_i, Mz_i, Fx_j, Fy_j, Fz_j, Mx_j, My_j, Mz_j],
    consistent with the element’s local displacement ordering
    [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2].
    convention: tension and positive bending act in the positive local directions.
    """
    import numpy as np
    try:
        E = float(ele_info['E'])
        nu = float(ele_info['nu'])
        A = float(ele_info['A'])
        I_y = float(ele_info['I_y'])
        I_z = float(ele_info['I_z'])
        J = float(ele_info['J'])
    except Exception as e:
        raise ValueError("ele_info must contain keys 'E','nu','A','I_y','I_z','J'") from e
    xi_f = float(xi)
    yi_f = float(yi)
    zi_f = float(zi)
    xj_f = float(xj)
    yj_f = float(yj)
    zj_f = float(zj)
    dx = xj_f - xi_f
    dy = yj_f - yi_f
    dz = zj_f - zi_f
    L = np.sqrt(dx * dx + dy * dy + dz * dz)
    if L <= 0.0 or not np.isfinite(L):
        raise ValueError('Beam length is zero or invalid.')
    ex = np.array([dx, dy, dz], dtype=float) / L
    local_z_candidate = ele_info.get('local_z', None)
    if local_z_candidate is None:
        global_z = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(np.dot(ex, global_z)) > 0.999999:
            a = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            a = global_z
    else:
        a = np.array(local_z_candidate, dtype=float)
        if a.shape != (3,):
            raise ValueError("ele_info['local_z'] must be array-like with shape (3,).")
    proj = np.dot(a, ex) * ex
    temp = a - proj
    norm_temp = np.linalg.norm(temp)
    if norm_temp <= 1e-12:
        raise ValueError('Provided local_z is parallel to the beam axis or invalid.')
    ez = temp / norm_temp
    ey = np.cross(ez, ex)
    ey_norm = np.linalg.norm(ey)
    if ey_norm <= 1e-12:
        raise ValueError('Failed to compute local y-axis; input orientation invalid.')
    ey = ey / ey_norm
    Lambda = np.vstack((ex, ey, ez)).astype(float)
    T = np.zeros((12, 12), dtype=float)
    T[0:3, 0:3] = Lambda
    T[3:6, 3:6] = Lambda
    T[6:9, 6:9] = Lambda
    T[9:12, 9:12] = Lambda
    u_g = np.asarray(u_dofs_global, dtype=float).reshape(-1)
    if u_g.size != 12:
        raise ValueError('u_dofs_global must have 12 entries.')
    u_local = T.dot(u_g)
    K = np.zeros((12, 12), dtype=float)
    EA_L = E * A / L
    K[0, 0] = EA_L
    K[0, 6] = -EA_L
    K[6, 0] = -EA_L
    K[6, 6] = EA_L
    G = E / (2.0 * (1.0 + nu))
    GJ_L = G * J / L
    K[3, 3] = GJ_L
    K[3, 9] = -GJ_L
    K[9, 3] = -GJ_L
    K[9, 9] = GJ_L
    EI_z = E * I_z
    factor_z = EI_z / L ** 3
    kz = factor_z * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L * L, -6.0 * L, 2.0 * L * L], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L * L, -6.0 * L, 4.0 * L * L]], dtype=float)
    idx_z = [1, 5, 7, 11]
    for i in range(4):
        for j in range(4):
            K[idx_z[i], idx_z[j]] += kz[i, j]
    EI_y = E * I_y
    factor_y = EI_y / L ** 3
    ky = factor_y * np.array([[12.0, -6.0 * L, -12.0, -6.0 * L], [-6.0 * L, 4.0 * L * L, 6.0 * L, 2.0 * L * L], [-12.0, 6.0 * L, 12.0, 6.0 * L], [-6.0 * L, 2.0 * L * L, 6.0 * L, 4.0 * L * L]], dtype=float)
    idx_y = [2, 4, 8, 10]
    for i in range(4):
        for j in range(4):
            K[idx_y[i], idx_y[j]] += ky[i, j]
    load_dofs_local = K.dot(u_local)
    return load_dofs_local.reshape(12)
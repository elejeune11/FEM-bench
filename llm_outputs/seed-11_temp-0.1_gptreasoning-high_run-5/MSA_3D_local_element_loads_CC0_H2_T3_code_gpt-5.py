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
    EPS_LEN = 1e-14
    EPS_PAR = 1e-10
    u_g = np.asarray(u_dofs_global, dtype=float).reshape(-1)
    if u_g.size != 12:
        raise ValueError('u_dofs_global must be an array-like of length 12.')
    dx = float(xj) - float(xi)
    dy = float(yj) - float(yi)
    dz = float(zj) - float(zi)
    L = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    if not np.isfinite(L) or L <= EPS_LEN:
        raise ValueError('Element length is zero or invalid.')
    ex = np.array([dx, dy, dz], dtype=float) / L
    local_z_ref = ele_info.get('local_z', None)
    if local_z_ref is None:
        if abs(ex[2]) >= 1.0 - EPS_PAR:
            z_ref = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            z_ref = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        z_ref = np.asarray(local_z_ref, dtype=float).reshape(-1)
        if z_ref.size != 3:
            raise ValueError('local_z must be an array-like of length 3.')
        if not np.all(np.isfinite(z_ref)):
            raise ValueError('local_z contains non-finite values.')
    nz = np.linalg.norm(z_ref)
    if nz <= EPS_LEN:
        raise ValueError('local_z must be a non-zero vector.')
    z_unit = z_ref / nz
    if abs(np.dot(z_unit, ex)) >= 1.0 - EPS_PAR:
        raise ValueError('local_z is invalid (parallel to element axis).')
    ey = np.cross(z_unit, ex)
    ny = np.linalg.norm(ey)
    if ny <= EPS_LEN:
        raise ValueError('Failed to construct a valid local coordinate system from local_z.')
    ey = ey / ny
    ez = np.cross(ex, ey)
    nz2 = np.linalg.norm(ez)
    if nz2 <= EPS_LEN:
        raise ValueError('Failed to construct a valid local coordinate system.')
    ez = ez / nz2
    R = np.vstack((ex, ey, ez))
    T = np.zeros((12, 12), dtype=float)
    T[0:3, 0:3] = R
    T[3:6, 3:6] = R
    T[6:9, 6:9] = R
    T[9:12, 9:12] = R
    u_l = T @ u_g
    E = float(ele_info['E'])
    nu = float(ele_info['nu'])
    A = float(ele_info['A'])
    I_y = float(ele_info['I_y'])
    I_z = float(ele_info['I_z'])
    J = float(ele_info['J'])
    G = E / (2.0 * (1.0 + nu))
    EA = E * A
    EIy = E * I_y
    EIz = E * I_z
    GJ = G * J
    L2 = L * L
    L3 = L2 * L
    K = np.zeros((12, 12), dtype=float)
    k_ax = EA / L
    K[0, 0] += k_ax
    K[0, 6] += -k_ax
    K[6, 0] += -k_ax
    K[6, 6] += k_ax
    k_to = GJ / L
    K[3, 3] += k_to
    K[3, 9] += -k_to
    K[9, 3] += -k_to
    K[9, 9] += k_to
    if EIz != 0.0:
        kbz = EIz / L3 * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]], dtype=float)
        idx_bz = [1, 5, 7, 11]
        for a in range(4):
            for b in range(4):
                K[idx_bz[a], idx_bz[b]] += kbz[a, b]
    if EIy != 0.0:
        kby = EIy / L3 * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]], dtype=float)
        idx_by = [2, 4, 8, 10]
        for a in range(4):
            for b in range(4):
                K[idx_by[a], idx_by[b]] += kby[a, b]
    load_dofs_local = K @ u_l
    return load_dofs_local
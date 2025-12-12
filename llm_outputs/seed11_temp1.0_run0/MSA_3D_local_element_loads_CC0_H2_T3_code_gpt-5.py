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
        Iy = float(ele_info['I_y'])
        Iz = float(ele_info['I_z'])
        J = float(ele_info['J'])
    except Exception as e:
        raise ValueError("ele_info must contain 'E', 'nu', 'A', 'I_y', 'I_z', and 'J' as numeric values.") from e
    pi = np.array([xi, yi, zi], dtype=float)
    pj = np.array([xj, yj, zj], dtype=float)
    d = pj - pi
    L = float(np.linalg.norm(d))
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError('Element length must be positive and finite.')
    ex = d / L
    z_ref = ele_info.get('local_z', None)
    tol = 1e-12
    if z_ref is None:
        global_z = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(ex, global_z)) >= 1.0 - 1e-08:
            z_ref = np.array([0.0, 1.0, 0.0])
        else:
            z_ref = global_z
    else:
        z_ref = np.asarray(z_ref, dtype=float).reshape(-1)
        if z_ref.size != 3 or not np.all(np.isfinite(z_ref)):
            raise ValueError('local_z must be a finite 3-component vector.')
        if np.linalg.norm(z_ref) <= tol:
            raise ValueError('local_z must be a non-zero vector.')
    cr = np.cross(z_ref, ex)
    n_cr = np.linalg.norm(cr)
    if n_cr <= 1e-10:
        raise ValueError('Provided local_z is parallel (or nearly parallel) to the element axis.')
    ey = cr / n_cr
    ez = np.cross(ex, ey)
    R = np.vstack((ex, ey, ez))
    u_g = np.asarray(u_dofs_global, dtype=float).reshape(12)
    ui_t = u_g[0:3]
    ui_r = u_g[3:6]
    uj_t = u_g[6:9]
    uj_r = u_g[9:12]
    u_l = np.concatenate([R @ ui_t, R @ ui_r, R @ uj_t, R @ uj_r], axis=0)
    G = E / (2.0 * (1.0 + nu))
    K = np.zeros((12, 12), dtype=float)
    k_ax = E * A / L
    K[0, 0] += k_ax
    K[0, 6] += -k_ax
    K[6, 0] += -k_ax
    K[6, 6] += k_ax
    k_tor = G * J / L
    K[3, 3] += k_tor
    K[3, 9] += -k_tor
    K[9, 3] += -k_tor
    K[9, 9] += k_tor
    c1 = 12.0 * E * Iz / L ** 3
    c2 = 6.0 * E * Iz / L ** 2
    c3 = 4.0 * E * Iz / L
    c4 = 2.0 * E * Iz / L
    dofs_v = [1, 5, 7, 11]
    K[dofs_v[0], dofs_v[0]] += c1
    K[dofs_v[0], dofs_v[1]] += c2
    K[dofs_v[0], dofs_v[2]] += -c1
    K[dofs_v[0], dofs_v[3]] += c2
    K[dofs_v[1], dofs_v[0]] += c2
    K[dofs_v[1], dofs_v[1]] += c3
    K[dofs_v[1], dofs_v[2]] += -c2
    K[dofs_v[1], dofs_v[3]] += c4
    K[dofs_v[2], dofs_v[0]] += -c1
    K[dofs_v[2], dofs_v[1]] += -c2
    K[dofs_v[2], dofs_v[2]] += c1
    K[dofs_v[2], dofs_v[3]] += -c2
    K[dofs_v[3], dofs_v[0]] += c2
    K[dofs_v[3], dofs_v[1]] += c4
    K[dofs_v[3], dofs_v[2]] += -c2
    K[dofs_v[3], dofs_v[3]] += c3
    b1 = 12.0 * E * Iy / L ** 3
    b2 = 6.0 * E * Iy / L ** 2
    b3 = 4.0 * E * Iy / L
    b4 = 2.0 * E * Iy / L
    dofs_w = [2, 4, 8, 10]
    K[dofs_w[0], dofs_w[0]] += b1
    K[dofs_w[0], dofs_w[1]] += b2
    K[dofs_w[0], dofs_w[2]] += -b1
    K[dofs_w[0], dofs_w[3]] += b2
    K[dofs_w[1], dofs_w[0]] += b2
    K[dofs_w[1], dofs_w[1]] += b3
    K[dofs_w[1], dofs_w[2]] += -b2
    K[dofs_w[1], dofs_w[3]] += b4
    K[dofs_w[2], dofs_w[0]] += -b1
    K[dofs_w[2], dofs_w[1]] += -b2
    K[dofs_w[2], dofs_w[2]] += b1
    K[dofs_w[2], dofs_w[3]] += -b2
    K[dofs_w[3], dofs_w[0]] += b2
    K[dofs_w[3], dofs_w[1]] += b4
    K[dofs_w[3], dofs_w[2]] += -b2
    K[dofs_w[3], dofs_w[3]] += b3
    load_local = K @ u_l
    return load_local
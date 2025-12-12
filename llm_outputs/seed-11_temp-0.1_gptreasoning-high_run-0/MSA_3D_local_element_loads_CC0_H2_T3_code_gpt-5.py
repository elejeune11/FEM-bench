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
    E = float(ele_info['E'])
    nu = float(ele_info['nu'])
    A = float(ele_info['A'])
    I_y = float(ele_info['I_y'])
    I_z = float(ele_info['I_z'])
    J = float(ele_info['J'])
    local_z_in = ele_info.get('local_z', None)
    pi = np.array([xi, yi, zi], dtype=float)
    pj = np.array([xj, yj, zj], dtype=float)
    t = pj - pi
    L = np.linalg.norm(t)
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError('Beam length is zero or invalid.')
    ex = t / L
    if local_z_in is None:
        z_ref = np.array([0.0, 0.0, 1.0], dtype=float)
        if np.linalg.norm(np.cross(ex, z_ref)) < 1e-12:
            z_ref = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        z_ref = np.asarray(local_z_in, dtype=float).reshape(-1)
        if z_ref.size != 3 or not np.all(np.isfinite(z_ref)):
            raise ValueError('Invalid local_z: must be a 3-component finite vector.')
        nrm = np.linalg.norm(z_ref)
        if nrm <= 0.0:
            raise ValueError('Invalid local_z: zero vector.')
        z_ref = z_ref / nrm
        if np.linalg.norm(np.cross(ex, z_ref)) < 1e-12:
            raise ValueError('Invalid local_z: parallel to element axis.')
    ey_temp = np.cross(z_ref, ex)
    ny = np.linalg.norm(ey_temp)
    if ny <= 1e-16:
        raise ValueError('Invalid local_z: leads to degenerate local axes.')
    ey = ey_temp / ny
    ez = np.cross(ex, ey)
    ez = ez / np.linalg.norm(ez)
    R = np.vstack((ex, ey, ez))
    T = np.zeros((12, 12), dtype=float)
    T[0:3, 0:3] = R
    T[3:6, 3:6] = R
    T[6:9, 6:9] = R
    T[9:12, 9:12] = R
    u_g = np.asarray(u_dofs_global, dtype=float).reshape(-1)
    if u_g.size != 12:
        raise ValueError('u_dofs_global must be array-like of shape (12,).')
    u_l = T @ u_g
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
    EIz = E * I_z
    c1 = 12.0 * EIz / L ** 3
    c2 = 6.0 * EIz / L ** 2
    c3 = 4.0 * EIz / L
    c4 = 2.0 * EIz / L
    (v1, thz1, v2, thz2) = (1, 5, 7, 11)
    K[v1, v1] += c1
    K[v1, thz1] += c2
    K[v1, v2] += -c1
    K[v1, thz2] += c2
    K[thz1, v1] += c2
    K[thz1, thz1] += c3
    K[thz1, v2] += -c2
    K[thz1, thz2] += c4
    K[v2, v1] += -c1
    K[v2, thz1] += -c2
    K[v2, v2] += c1
    K[v2, thz2] += -c2
    K[thz2, v1] += c2
    K[thz2, thz1] += c4
    K[thz2, v2] += -c2
    K[thz2, thz2] += c3
    EIy = E * I_y
    d1 = 12.0 * EIy / L ** 3
    d2 = 6.0 * EIy / L ** 2
    d3 = 4.0 * EIy / L
    d4 = 2.0 * EIy / L
    (w1, thy1, w2, thy2) = (2, 4, 8, 10)
    K[w1, w1] += d1
    K[w1, thy1] += -d2
    K[w1, w2] += -d1
    K[w1, thy2] += -d2
    K[thy1, w1] += -d2
    K[thy1, thy1] += d3
    K[thy1, w2] += d2
    K[thy1, thy2] += d4
    K[w2, w1] += -d1
    K[w2, thy1] += d2
    K[w2, w2] += d1
    K[w2, thy2] += d2
    K[thy2, w1] += -d2
    K[thy2, thy1] += d4
    K[thy2, w2] += d2
    K[thy2, thy2] += d3
    load_dofs_local = K @ u_l
    return load_dofs_local.reshape(12)
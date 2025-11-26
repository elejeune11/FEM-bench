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
    E = ele_info['E']
    nu = ele_info['nu']
    A = ele_info['A']
    Iy = ele_info['I_y']
    Iz = ele_info['I_z']
    J = ele_info['J']
    dx = xj - xi
    dy = yj - yi
    dz = zj - zi
    L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    if L == 0:
        raise ValueError('Element length is zero.')
    vec_u = np.array([dx, dy, dz]) / L
    if 'local_z' in ele_info and ele_info['local_z'] is not None:
        vec_ref = np.array(ele_info['local_z'], dtype=float)
        norm_ref = np.linalg.norm(vec_ref)
        if norm_ref == 0:
            raise ValueError('Provided local_z has zero length.')
        vec_ref = vec_ref / norm_ref
    else:
        global_z = np.array([0.0, 0.0, 1.0])
        if np.linalg.norm(np.cross(vec_u, global_z)) < 1e-08:
            vec_ref = np.array([0.0, 1.0, 0.0])
        else:
            vec_ref = global_z
    if np.linalg.norm(np.cross(vec_u, vec_ref)) < 1e-08:
        raise ValueError('local_z is parallel to the beam axis.')
    vec_v = np.cross(vec_ref, vec_u)
    vec_v = vec_v / np.linalg.norm(vec_v)
    vec_w = np.cross(vec_u, vec_v)
    vec_w = vec_w / np.linalg.norm(vec_w)
    R = np.vstack([vec_u, vec_v, vec_w])
    u_dofs = np.array(u_dofs_global)
    u_local = np.zeros(12)
    u_local[0:3] = R @ u_dofs[0:3]
    u_local[3:6] = R @ u_dofs[3:6]
    u_local[6:9] = R @ u_dofs[6:9]
    u_local[9:12] = R @ u_dofs[9:12]
    G = E / (2 * (1 + nu))
    k_local = np.zeros((12, 12))
    X = E * A / L
    T = G * J / L
    Y1 = 12 * E * Iz / L ** 3
    Y2 = 6 * E * Iz / L ** 2
    Y3 = 4 * E * Iz / L
    Y4 = 2 * E * Iz / L
    Z1 = 12 * E * Iy / L ** 3
    Z2 = 6 * E * Iy / L ** 2
    Z3 = 4 * E * Iy / L
    Z4 = 2 * E * Iy / L
    k_local[0, 0] = X
    k_local[0, 6] = -X
    k_local[6, 0] = -X
    k_local[6, 6] = X
    k_local[3, 3] = T
    k_local[3, 9] = -T
    k_local[9, 3] = -T
    k_local[9, 9] = T
    idx_y = [1, 5, 7, 11]
    k_sub_y = np.array([[Y1, Y2, -Y1, Y2], [Y2, Y3, -Y2, Y4], [-Y1, -Y2, Y1, -Y2], [Y2, Y4, -Y2, Y3]])
    for i in range(4):
        for j in range(4):
            k_local[idx_y[i], idx_y[j]] = k_sub_y[i, j]
    idx_z = [2, 4, 8, 10]
    k_sub_z = np.array([[Z1, -Z2, -Z1, -Z2], [-Z2, Z3, Z2, Z4], [-Z1, Z2, Z1, Z2], [-Z2, Z4, Z2, Z3]])
    for i in range(4):
        for j in range(4):
            k_local[idx_z[i], idx_z[j]] = k_sub_z[i, j]
    load_dofs_local = k_local @ u_local
    return load_dofs_local
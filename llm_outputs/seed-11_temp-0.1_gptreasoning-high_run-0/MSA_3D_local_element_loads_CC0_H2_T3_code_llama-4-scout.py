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
    beam_axis = np.array([xj - xi, yj - yi, zj - zi])
    beam_length = np.linalg.norm(beam_axis)
    if beam_length == 0:
        raise ValueError('Beam length cannot be zero')
    beam_axis = beam_axis / beam_length
    if 'local_z' not in ele_info or ele_info['local_z'] is None:
        if np.isclose(beam_axis[0], 0) and np.isclose(beam_axis[1], 0):
            local_z = np.array([0, 1, 0])
        else:
            local_z = np.array([0, 0, 1])
    else:
        local_z = np.array(ele_info['local_z'])
        if np.linalg.norm(local_z) == 0 or np.isclose(np.dot(local_z, beam_axis), 1):
            raise ValueError('Invalid local z-axis orientation')
    local_z = local_z / np.linalg.norm(local_z)
    local_x = beam_axis
    local_y = np.cross(local_z, local_x)
    local_y = local_y / np.linalg.norm(local_y)
    T_local_to_global = np.array([[local_x[0], local_y[0], local_z[0], 0, 0, 0, 0, 0, 0, 0, 0, 0], [local_x[1], local_y[1], local_z[1], 0, 0, 0, 0, 0, 0, 0, 0, 0], [local_x[2], local_y[2], local_z[2], 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, local_x[0], local_y[0], local_z[0], 0, 0, 0, 0, 0, 0], [0, 0, 0, local_x[1], local_y[1], local_z[1], 0, 0, 0, 0, 0, 0], [0, 0, 0, local_x[2], local_y[2], local_z[2], 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, local_x[0], local_y[0], local_z[0], 0, 0, 0], [0, 0, 0, 0, 0, 0, local_x[1], local_y[1], local_z[1], 0, 0, 0], [0, 0, 0, 0, 0, 0, local_x[2], local_y[2], local_z[2], 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, local_x[0], local_y[0], local_z[0]], [0, 0, 0, 0, 0, 0, 0, 0, 0, local_x[1], local_y[1], local_z[1]], [0, 0, 0, 0, 0, 0, 0, 0, 0, local_x[2], local_y[2], local_z[2]]])
    T_global_to_local = np.linalg.inv(T_local_to_global)
    u_dofs_local = np.dot(T_global_to_local, u_dofs_global)
    E = ele_info['E']
    nu = ele_info['nu']
    A = ele_info['A']
    I_y = ele_info['I_y']
    I_z = ele_info['I_z']
    J = ele_info['J']
    k_local = np.array([[E * A / beam_length, 0, 0, 0, 0, 0, -E * A / beam_length, 0, 0, 0, 0, 0], [0, 12 * E * I_z / beam_length ** 3, 0, 0, 0, 6 * E * I_z / beam_length ** 2, 0, -12 * E * I_z / beam_length ** 3, 0, 0, 0, 0, 6 * E * I_z / beam_length ** 2], [0, 0, 12 * E * I_y / beam_length ** 3, 0, -6 * E * I_y / beam_length ** 2, 0, 0, 0, -12 * E * I_y / beam_length ** 3, 0, -6 * E * I_y / beam_length ** 2, 0], [0, 0, 0, E * J / beam_length, 0, 0, 0, 0, 0, 0, E * J / beam_length, 0, 0], [0, 0, -6 * E * I_y / beam_length ** 2, 0, 4 * E * I_y / beam_length, 0, 0, 0, 6 * E * I_y / beam_length ** 2, 0, 2 * E * I_y / beam_length, 0], [0, 6 * E * I_z / beam_length ** 2, 0, 0, 0, 4 * E * I_z / beam_length, 0, -6 * E * I_z / beam_length ** 2, 0, 0, 0, 2 * E * I_z / beam_length], [-E * A / beam_length, 0, 0, 0, 0, 0, E * A / beam_length, 0, 0, 0, 0, 0], [0, -12 * E * I_z / beam_length ** 3, 0, 0, 0, -6 * E * I_z / beam_length ** 2, 0, 12 * E * I_z / beam_length ** 3, 0, 0, 0, 0, -6 * E * I_z / beam_length ** 2], [0, 0, -12 * E * I_y / beam_length ** 3, 0, 6 * E * I_y / beam_length ** 2, 0, 0, 0, 12 * E * I_y / beam_length ** 3, 0, 6 * E * I_y / beam_length ** 2, 0], [0, 0, 0, E * J / beam_length, 0, 0, 0, 0, 0, 0, -E * J / beam_length, 0, 0], [0, 0, 6 * E * I_y / beam_length ** 2, 0, 2 * E * I_y / beam_length, 0, 0, 0, -6 * E * I_y / beam_length ** 2, 0, 4 * E * I_y / beam_length, 0], [0, 6 * E * I_z / beam_length ** 2, 0, 0, 0, 2 * E * I_z / beam_length, 0, -6 * E * I_z / beam_length ** 2, 0, 0, 0, 4 * E * I_z / beam_length]])
    load_dofs_local = np.dot(k_local, u_dofs_local)
    return load_dofs_local
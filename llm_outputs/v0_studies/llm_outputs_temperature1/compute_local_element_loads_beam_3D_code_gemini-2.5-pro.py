def compute_local_element_loads_beam_3D(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global):
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
      to the provided displacement state, not externally applied loads.
    Support Functions Used
    ----------------------
        Computes the 12x12 transformation matrix (Gamma) relating local and global coordinate systems
        for a 3D beam element. Ensures orthonormal local axes and validates the reference vector.
        Returns the 12x12 local elastic stiffness matrix for a 3D Euler-Bernoulli beam aligned with the
        local x-axis, capturing axial, bending, and torsional stiffness.
    """

    def local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J):
        k = np.zeros((12, 12))
        EA_L = E * A / L
        GJ_L = E * J / (2.0 * (1.0 + nu) * L)
        EIz_L = E * Iz
        EIy_L = E * Iy
        k[0, 0] = k[6, 6] = EA_L
        k[0, 6] = k[6, 0] = -EA_L
        k[3, 3] = k[9, 9] = GJ_L
        k[3, 9] = k[9, 3] = -GJ_L
        k[1, 1] = k[7, 7] = 12.0 * EIz_L / L ** 3
        k[1, 7] = k[7, 1] = -12.0 * EIz_L / L ** 3
        k[1, 5] = k[5, 1] = k[1, 11] = k[11, 1] = 6.0 * EIz_L / L ** 2
        k[5, 7] = k[7, 5] = k[7, 11] = k[11, 7] = -6.0 * EIz_L / L ** 2
        k[5, 5] = k[11, 11] = 4.0 * EIz_L / L
        k[5, 11] = k[11, 5] = 2.0 * EIz_L / L
        k[2, 2] = k[8, 8] = 12.0 * EIy_L / L ** 3
        k[2, 8] = k[8, 2] = -12.0 * EIy_L / L ** 3
        k[2, 4] = k[4, 2] = k[2, 10] = k[10, 2] = -6.0 * EIy_L / L ** 2
        k[4, 8] = k[8, 4] = k[8, 10] = k[10, 8] = 6.0 * EIy_L / L ** 2
        k[4, 4] = k[10, 10] = 4.0 * EIy_L / L
        k[4, 10] = k[10, 4] = 2.0 * EIy_L / L
        return k

    def beam_transformation_matrix_3D(x1, y1, z1, x2, y2, z2, ref_vec: Optional[np.ndarray]):
        (dx, dy, dz) = (x2 - x1, y2 - y1, z2 - z1)
        L = np.sqrt(dx * dx + dy * dy + dz * dz)
        if np.isclose(L, 0.0):
            raise ValueError('Beam length is zero.')
        ex = np.array([dx, dy, dz]) / L
        if ref_vec is None:
            ref_vec = np.array([0.0, 0.0, 1.0]) if not (np.isclose(ex[0], 0) and np.isclose(ex[1], 0)) else np.array([0.0, 1.0, 0.0])
        else:
            ref_vec = np.asarray(ref_vec, dtype=float)
            if ref_vec.shape != (3,):
                raise ValueError('local_z/reference_vector must be length‑3.')
            if not np.isclose(np.linalg.norm(ref_vec), 1.0):
                raise ValueError('reference_vector must be unit length.')
            if np.isclose(np.linalg.norm(np.cross(ref_vec, ex)), 0.0):
                raise ValueError('reference_vector parallel to beam axis.')
        ey = np.cross(ref_vec, ex)
        ey /= np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        gamma = np.vstack((ex, ey, ez))
        return np.kron(np.eye(4), gamma)
    E = ele_info['E']
    nu = ele_info['nu']
    A = ele_info['A']
    Iy = ele_info['I_y']
    Iz = ele_info['I_z']
    J = ele_info['J']
    local_z_vec = ele_info.get('local_z')
    L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
    Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ref_vec=local_z_vec)
    k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
    u_dofs_global_np = np.asarray(u_dofs_global)
    u_dofs_local = Gamma @ u_dofs_global_np
    load_dofs_local = k_local @ u_dofs_local
    return load_dofs_local
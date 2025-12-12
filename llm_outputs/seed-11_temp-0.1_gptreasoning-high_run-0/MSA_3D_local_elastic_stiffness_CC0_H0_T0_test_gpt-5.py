def test_local_stiffness_3D_beam(fcn):
    """
    Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    """
    E = 210000000000.0
    nu = 0.29
    A = 0.02
    L = 2.0
    Iy = 8e-06
    Iz = 5e-06
    J = 1.2e-05
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert isinstance(K, np.ndarray)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T, rtol=1e-12, atol=1e-12)
    s = np.linalg.svd(K, compute_uv=False)
    tol = max(1e-12 * s.max(), 1e-14)
    nullity = np.sum(s < tol)
    assert nullity == 6
    k_ax = E * A / L
    axial_indices = [0, 6]
    K_axial = K[np.ix_(axial_indices, axial_indices)]
    K_axial_expected = np.array([[1, -1], [-1, 1]], dtype=float) * k_ax
    assert np.allclose(K_axial, K_axial_expected, rtol=1e-12, atol=1e-12)
    other_cols = [i for i in range(12) if i not in axial_indices]
    assert np.allclose(K[np.ix_(axial_indices, other_cols)], 0.0, rtol=0, atol=1e-12)
    assert np.allclose(K[np.ix_(other_cols, axial_indices)], 0.0, rtol=0, atol=1e-12)
    G = E / (2 * (1 + nu))
    k_tor = G * J / L
    torsion_indices = [3, 9]
    K_torsion = K[np.ix_(torsion_indices, torsion_indices)]
    K_torsion_expected = np.array([[1, -1], [-1, 1]], dtype=float) * k_tor
    assert np.allclose(K_torsion, K_torsion_expected, rtol=1e-12, atol=1e-12)
    other_cols_t = [i for i in range(12) if i not in torsion_indices]
    assert np.allclose(K[np.ix_(torsion_indices, other_cols_t)], 0.0, rtol=0, atol=1e-12)
    assert np.allclose(K[np.ix_(other_cols_t, torsion_indices)], 0.0, rtol=0, atol=1e-12)
    a_z = 12 * E * Iz / L ** 3
    b_z = 6 * E * Iz / L ** 2
    c_z = 4 * E * Iz / L
    d_z = 2 * E * Iz / L
    bending_z_indices = [1, 5, 7, 11]
    K_bend_z = K[np.ix_(bending_z_indices, bending_z_indices)]
    K_bend_z_expected = np.array([[a_z, b_z, -a_z, b_z], [b_z, c_z, -b_z, d_z], [-a_z, -b_z, a_z, -b_z], [b_z, d_z, -b_z, c_z]], dtype=float)
    assert np.allclose(K_bend_z, K_bend_z_expected, rtol=1e-12, atol=1e-09)
    a_y = 12 * E * Iy / L ** 3
    b_y = 6 * E * Iy / L ** 2
    c_y = 4 * E * Iy / L
    d_y = 2 * E * Iy / L
    bending_y_indices = [2, 4, 8, 10]
    K_bend_y = K[np.ix_(bending_y_indices, bending_y_indices)]
    K_bend_y_expected = np.array([[a_y, -b_y, -a_y, -b_y], [-b_y, c_y, b_y, d_y], [-a_y, b_y, a_y, b_y], [-b_y, d_y, b_y, c_y]], dtype=float)
    assert np.allclose(K_bend_y, K_bend_y_expected, rtol=1e-12, atol=1e-09)
    cross = K[np.ix_(bending_z_indices, bending_y_indices)]
    assert np.allclose(cross, 0.0, rtol=0, atol=1e-12)
    cross_T = K[np.ix_(bending_y_indices, bending_z_indices)]
    assert np.allclose(cross_T, 0.0, rtol=0, atol=1e-12)

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """
    Cantilever beam tip-load tests using a single 3D Euler-Bernoulli element:
    """
    E = 210000000000.0
    nu = 0.29
    A = 0.02
    L = 2.0
    Iy = 8e-06
    Iz = 5e-06
    J = 1.2e-05
    K = fcn(E, nu, A, L, Iy, Iz, J)
    fixed = [0, 1, 2, 3, 4, 5]
    free = [6, 7, 8, 9, 10, 11]

    def solve_with_load(F):
        K_ff = K[np.ix_(free, free)]
        d_free = np.linalg.solve(K_ff, F[free])
        d = np.zeros(12)
        d[free] = d_free
        return d
    P = 1000.0
    Fz = np.zeros(12)
    Fz[8] = P
    d = solve_with_load(Fz)
    w_tip_expected = P * L ** 3 / (3 * E * Iy)
    assert np.isclose(d[8], w_tip_expected, rtol=1e-10, atol=1e-12)
    Fy = np.zeros(12)
    Fy[7] = P
    d = solve_with_load(Fy)
    v_tip_expected = P * L ** 3 / (3 * E * Iz)
    assert np.isclose(d[7], v_tip_expected, rtol=1e-10, atol=1e-12)
    Fx = np.zeros(12)
    Fx[6] = P
    d = solve_with_load(Fx)
    u_tip_expected = P * L / (E * A)
    assert np.isclose(d[6], u_tip_expected, rtol=1e-12, atol=1e-15)
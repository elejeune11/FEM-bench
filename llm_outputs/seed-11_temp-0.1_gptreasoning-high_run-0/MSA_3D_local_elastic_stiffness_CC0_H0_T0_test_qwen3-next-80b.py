def test_local_stiffness_3D_beam(fcn):
    (E, nu, A, L, Iy, Iz, J) = (200000000000.0, 0.3, 0.01, 2.0, 1e-06, 1e-06, 5e-07)
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T)
    assert np.isclose(np.linalg.det(K), 0, atol=1e-10)
    k_axial = E * A / L
    k_torsion = E * J / (2 * (1 + nu)) / L
    k_bend_y = E * Iy / L ** 3
    k_bend_z = E * Iz / L ** 3
    assert np.isclose(K[0, 0], k_axial)
    assert np.isclose(K[6, 6], k_axial)
    assert np.isclose(K[0, 6], -k_axial)
    assert np.isclose(K[3, 3], k_torsion)
    assert np.isclose(K[9, 9], k_torsion)
    assert np.isclose(K[3, 9], -k_torsion)
    assert np.isclose(K[1, 1], 12 * k_bend_z)
    assert np.isclose(K[5, 5], 4 * k_bend_z * L ** 2)
    assert np.isclose(K[1, 5], 6 * k_bend_z * L)
    assert np.isclose(K[7, 7], 12 * k_bend_z)
    assert np.isclose(K[11, 11], 4 * k_bend_z * L ** 2)
    assert np.isclose(K[7, 11], 6 * k_bend_z * L)
    assert np.isclose(K[1, 7], -12 * k_bend_z)
    assert np.isclose(K[1, 11], 6 * k_bend_z * L)
    assert np.isclose(K[5, 7], -6 * k_bend_z * L)
    assert np.isclose(K[5, 11], -2 * k_bend_z * L ** 2)
    assert np.isclose(K[2, 2], 12 * k_bend_y)
    assert np.isclose(K[4, 4], 4 * k_bend_y * L ** 2)
    assert np.isclose(K[2, 4], 6 * k_bend_y * L)
    assert np.isclose(K[8, 8], 12 * k_bend_y)
    assert np.isclose(K[10, 10], 4 * k_bend_y * L ** 2)
    assert np.isclose(K[8, 10], 6 * k_bend_y * L)
    assert np.isclose(K[2, 8], -12 * k_bend_y)
    assert np.isclose(K[2, 10], 6 * k_bend_y * L)
    assert np.isclose(K[4, 8], -6 * k_bend_y * L)
    assert np.isclose(K[4, 10], -2 * k_bend_y * L ** 2)

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    (E, nu, A, L, Iy, Iz, J) = (200000000000.0, 0.3, 0.01, 1.0, 1e-06, 1e-06, 5e-07)
    K = fcn(E, nu, A, L, Iy, Iz, J)
    F = np.zeros(12)
    F[2] = -1000
    F[1] = -1000
    F[0] = -1000
    fixed_dofs = [0, 1, 2, 3, 4, 5]
    free_dofs = [6, 7, 8, 9, 10, 11]
    K_free = K[np.ix_(free_dofs, free_dofs)]
    F_free = F[free_dofs]
    u_free = np.linalg.solve(K_free, F_free)
    u_z_analytical = -1000 * L ** 3 / (3 * E * Iz)
    u_y_analytical = -1000 * L ** 3 / (3 * E * Iy)
    u_x_analytical = -1000 * L / (E * A)
    assert np.isclose(u_free[2], u_z_analytical, rtol=1e-06)
    assert np.isclose(u_free[1], u_y_analytical, rtol=1e-06)
    assert np.isclose(u_free[0], u_x_analytical, rtol=1e-06)
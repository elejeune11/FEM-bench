def test_local_stiffness_3D_beam(fcn):
    """Comprehensive test for local_elastic_stiffness_matrix_3D_beam."""
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    L = 1.0
    Iy = 0.0001
    Iz = 0.0001
    J = 0.0001
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert K.shape == (12, 12)
    assert_array_almost_equal(K, K.T)
    nullspace = null_space(K)
    assert nullspace.shape[1] >= 6
    EA_L = E * A / L
    assert_array_almost_equal(K[0, 0], EA_L)
    assert_array_almost_equal(K[0, 6], -EA_L)
    GJ_L = E * J / (2 * (1 + nu) * L)
    assert_array_almost_equal(K[3, 3], GJ_L)
    assert_array_almost_equal(K[3, 9], -GJ_L)
    EI_L3 = 12 * E * Iz / L ** 3
    assert_array_almost_equal(K[1, 1], EI_L3)
    assert_array_almost_equal(K[1, 7], -EI_L3)

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """Test cantilever beam deflections against analytical solutions."""
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    L = 2.0
    Iy = 0.0001
    Iz = 0.0001
    J = 0.0001
    P = 1000.0
    K = fcn(E, nu, A, L, Iy, Iz, J)
    F_z = np.zeros(12)
    F_z[2] = P
    F_y = np.zeros(12)
    F_y[1] = P
    F_x = np.zeros(12)
    F_x[0] = P
    reduced_K = K[6:, 6:]
    d_z = np.linalg.solve(reduced_K, F_z[6:])
    analytical_z = P * L ** 3 / (3 * E * Iz)
    assert_array_almost_equal(d_z[2], analytical_z, decimal=6)
    d_y = np.linalg.solve(reduced_K, F_y[6:])
    analytical_y = P * L ** 3 / (3 * E * Iy)
    assert_array_almost_equal(d_y[1], analytical_y, decimal=6)
    d_x = np.linalg.solve(reduced_K, F_x[6:])
    analytical_x = P * L / (E * A)
    assert_array_almost_equal(d_x[0], analytical_x, decimal=6)
def test_local_stiffness_3D_beam(fcn):
    """
    Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    """
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
    null_vectors = null_space(K)
    assert null_vectors.shape[1] == 6
    assert_array_almost_equal(K[0, 0], E * A / L)
    assert_array_almost_equal(K[0, 6], -E * A / L)
    G = E / (2 * (1 + nu))
    assert_array_almost_equal(K[3, 3], G * J / L)
    assert_array_almost_equal(K[3, 9], -G * J / L)
    assert_array_almost_equal(K[1, 1], 12 * E * Iz / L ** 3)
    assert_array_almost_equal(K[1, 5], 6 * E * Iz / L ** 2)
    assert_array_almost_equal(K[5, 5], 4 * E * Iz / L)
    assert_array_almost_equal(K[2, 2], 12 * E * Iy / L ** 3)
    assert_array_almost_equal(K[2, 4], -6 * E * Iy / L ** 2)
    assert_array_almost_equal(K[4, 4], 4 * E * Iy / L)

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """
    Verify cantilever beam deflections match Euler-Bernoulli beam theory for:
    """
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    L = 2.0
    Iy = 0.0001
    Iz = 0.0001
    J = 0.0001
    K = fcn(E, nu, A, L, Iy, Iz, J)
    Kr = K[6:, 6:]
    P = 1000.0
    F = np.zeros(6)
    F[2] = P
    d = np.linalg.solve(Kr, F)
    analytical_z = P * L ** 3 / (3 * E * Iy)
    assert_array_almost_equal(d[2], analytical_z, decimal=6)
    F = np.zeros(6)
    F[1] = P
    d = np.linalg.solve(Kr, F)
    analytical_y = P * L ** 3 / (3 * E * Iz)
    assert_array_almost_equal(d[1], analytical_y, decimal=6)
    F = np.zeros(6)
    F[0] = P
    d = np.linalg.solve(Kr, F)
    analytical_x = P * L / (E * A)
    assert_array_almost_equal(d[0], analytical_x, decimal=6)
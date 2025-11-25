def test_local_stiffness_3D_beam(fcn):
    """
    Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    """
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    L = 2.0
    Iy = 8.33e-06
    Iz = 8.33e-06
    J = 1.67e-05
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T, rtol=1e-12)
    eigenvals = np.linalg.eigvals(K)
    eigenvals_sorted = np.sort(eigenvals)
    assert np.sum(np.abs(eigenvals_sorted[:6]) < 1e-10) == 6
    G = E / (2 * (1 + nu))
    expected_axial = E * A / L
    assert np.isclose(K[0, 0], expected_axial, rtol=1e-12)
    assert np.isclose(K[6, 6], expected_axial, rtol=1e-12)
    assert np.isclose(K[0, 6], -expected_axial, rtol=1e-12)
    expected_torsion = G * J / L
    assert np.isclose(K[3, 3], expected_torsion, rtol=1e-12)
    assert np.isclose(K[9, 9], expected_torsion, rtol=1e-12)
    assert np.isclose(K[3, 9], -expected_torsion, rtol=1e-12)
    expected_bend_v = 12 * E * Iz / L ** 3
    expected_bend_moment = 4 * E * Iz / L
    assert np.isclose(K[1, 1], expected_bend_v, rtol=1e-12)
    assert np.isclose(K[7, 7], expected_bend_v, rtol=1e-12)
    assert np.isclose(K[5, 5], expected_bend_moment, rtol=1e-12)
    assert np.isclose(K[11, 11], expected_bend_moment, rtol=1e-12)
    expected_bend_w = 12 * E * Iy / L ** 3
    expected_bend_moment_y = 4 * E * Iy / L
    assert np.isclose(K[2, 2], expected_bend_w, rtol=1e-12)
    assert np.isclose(K[8, 8], expected_bend_w, rtol=1e-12)
    assert np.isclose(K[4, 4], expected_bend_moment_y, rtol=1e-12)
    assert np.isclose(K[10, 10], expected_bend_moment_y, rtol=1e-12)

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """
    Apply a perpendicular point load in the z direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a perpendicular point load in the y direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a parallel point load in the x direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    """
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    L = 2.0
    Iy = 8.33e-06
    Iz = 8.33e-06
    J = 1.67e-05
    K = fcn(E, nu, A, L, Iy, Iz, J)
    K_free = K[6:12, 6:12]
    F_z = 1000.0
    force_vector = np.zeros(6)
    force_vector[2] = F_z
    displacements = np.linalg.solve(K_free, force_vector)
    computed_w2 = displacements[2]
    analytical_w2 = F_z * L ** 3 / (3 * E * Iy)
    assert np.isclose(computed_w2, analytical_w2, rtol=1e-10)
    F_y = 1000.0
    force_vector = np.zeros(6)
    force_vector[1] = F_y
    displacements = np.linalg.solve(K_free, force_vector)
    computed_v2 = displacements[1]
    analytical_v2 = F_y * L ** 3 / (3 * E * Iz)
    assert np.isclose(computed_v2, analytical_v2, rtol=1e-10)
    F_x = 1000.0
    force_vector = np.zeros(6)
    force_vector[0] = F_x
    displacements = np.linalg.solve(K_free, force_vector)
    computed_u2 = displacements[0]
    analytical_u2 = F_x * L / (E * A)
    assert np.isclose(computed_u2, analytical_u2, rtol=1e-10)
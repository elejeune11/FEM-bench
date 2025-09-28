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
    assert np.allclose(K, K.T)
    eigenvals = np.linalg.eigvals(K)
    zero_eigenvals = np.sum(np.abs(eigenvals) < 1e-10)
    assert zero_eigenvals == 6
    expected_axial = E * A / L
    assert np.isclose(K[0, 0], expected_axial)
    assert np.isclose(K[6, 6], expected_axial)
    assert np.isclose(K[0, 6], -expected_axial)
    G = E / (2 * (1 + nu))
    expected_torsion = G * J / L
    assert np.isclose(K[3, 3], expected_torsion)
    assert np.isclose(K[9, 9], expected_torsion)
    assert np.isclose(K[3, 9], -expected_torsion)
    expected_bending_z = 12 * E * Iz / L ** 3
    assert np.isclose(K[1, 1], expected_bending_z)
    assert np.isclose(K[7, 7], expected_bending_z)
    expected_bending_y = 12 * E * Iy / L ** 3
    assert np.isclose(K[2, 2], expected_bending_y)
    assert np.isclose(K[8, 8], expected_bending_y)

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
    F_z = np.zeros(6)
    F_z[2] = 1000.0
    u_z = np.linalg.solve(K_free, F_z)
    computed_deflection_z = u_z[2]
    analytical_deflection_z = 1000.0 * L ** 3 / (3 * E * Iy)
    assert np.isclose(computed_deflection_z, analytical_deflection_z, rtol=1e-10)
    F_y = np.zeros(6)
    F_y[1] = 1000.0
    u_y = np.linalg.solve(K_free, F_y)
    computed_deflection_y = u_y[1]
    analytical_deflection_y = 1000.0 * L ** 3 / (3 * E * Iz)
    assert np.isclose(computed_deflection_y, analytical_deflection_y, rtol=1e-10)
    F_x = np.zeros(6)
    F_x[0] = 1000.0
    u_x = np.linalg.solve(K_free, F_x)
    computed_extension_x = u_x[0]
    analytical_extension_x = 1000.0 * L / (E * A)
    assert np.isclose(computed_extension_x, analytical_extension_x, rtol=1e-10)
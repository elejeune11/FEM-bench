def test_local_stiffness_3D_beam(fcn):
    """Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    L = 2.0
    Iy = 1e-05
    Iz = 2e-05
    J = 1.5e-05
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert K.shape == (12, 12), 'Stiffness matrix should be 12x12'
    assert np.allclose(K, K.T, rtol=1e-10), 'Stiffness matrix should be symmetric'
    eigenvalues = np.linalg.eigvalsh(K)
    num_zero_eigenvalues = np.sum(np.abs(eigenvalues) < 1e-06 * np.max(np.abs(eigenvalues)))
    assert num_zero_eigenvalues == 6, f'Expected 6 zero eigenvalues for rigid body modes, got {num_zero_eigenvalues}'
    G = E / (2 * (1 + nu))
    axial_stiffness = E * A / L
    assert np.isclose(K[0, 0], axial_stiffness, rtol=1e-10), 'Axial stiffness K[0,0] incorrect'
    assert np.isclose(K[6, 6], axial_stiffness, rtol=1e-10), 'Axial stiffness K[6,6] incorrect'
    assert np.isclose(K[0, 6], -axial_stiffness, rtol=1e-10), 'Axial coupling K[0,6] incorrect'
    torsional_stiffness = G * J / L
    assert np.isclose(K[3, 3], torsional_stiffness, rtol=1e-10), 'Torsional stiffness K[3,3] incorrect'
    assert np.isclose(K[9, 9], torsional_stiffness, rtol=1e-10), 'Torsional stiffness K[9,9] incorrect'
    assert np.isclose(K[3, 9], -torsional_stiffness, rtol=1e-10), 'Torsional coupling K[3,9] incorrect'
    bending_z_stiffness = 12 * E * Iz / L ** 3
    assert np.isclose(K[1, 1], bending_z_stiffness, rtol=1e-10), 'Bending stiffness K[1,1] incorrect'
    assert np.isclose(K[7, 7], bending_z_stiffness, rtol=1e-10), 'Bending stiffness K[7,7] incorrect'
    bending_y_stiffness = 12 * E * Iy / L ** 3
    assert np.isclose(K[2, 2], bending_y_stiffness, rtol=1e-10), 'Bending stiffness K[2,2] incorrect'
    assert np.isclose(K[8, 8], bending_y_stiffness, rtol=1e-10), 'Bending stiffness K[8,8] incorrect'
    rotational_z_stiffness = 4 * E * Iz / L
    assert np.isclose(K[5, 5], rotational_z_stiffness, rtol=1e-10), 'Rotational stiffness K[5,5] incorrect'
    assert np.isclose(K[11, 11], rotational_z_stiffness, rtol=1e-10), 'Rotational stiffness K[11,11] incorrect'
    rotational_y_stiffness = 4 * E * Iy / L
    assert np.isclose(K[4, 4], rotational_y_stiffness, rtol=1e-10), 'Rotational stiffness K[4,4] incorrect'
    assert np.isclose(K[10, 10], rotational_y_stiffness, rtol=1e-10), 'Rotational stiffness K[10,10] incorrect'

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """Apply a perpendicular point load in the z direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a perpendicular point load in the y direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a parallel point load in the x direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    L = 2.0
    Iy = 1e-05
    Iz = 2e-05
    J = 1.5e-05
    K = fcn(E, nu, A, L, Iy, Iz, J)
    K_free = K[6:12, 6:12]
    P_z = 1000.0
    F_z = np.array([0, 0, P_z, 0, 0, 0])
    u_z = np.linalg.solve(K_free, F_z)
    w_analytical_z = P_z * L ** 3 / (3 * E * Iy)
    assert np.isclose(u_z[2], w_analytical_z, rtol=1e-06), f'Z-direction deflection mismatch: computed {u_z[2]}, expected {w_analytical_z}'
    theta_y_analytical = P_z * L ** 2 / (2 * E * Iy)
    assert np.isclose(u_z[4], -theta_y_analytical, rtol=1e-06), f'Rotation about y mismatch: computed {u_z[4]}, expected {-theta_y_analytical}'
    P_y = 1000.0
    F_y = np.array([0, P_y, 0, 0, 0, 0])
    u_y = np.linalg.solve(K_free, F_y)
    v_analytical_y = P_y * L ** 3 / (3 * E * Iz)
    assert np.isclose(u_y[1], v_analytical_y, rtol=1e-06), f'Y-direction deflection mismatch: computed {u_y[1]}, expected {v_analytical_y}'
    theta_z_analytical = P_y * L ** 2 / (2 * E * Iz)
    assert np.isclose(u_y[5], theta_z_analytical, rtol=1e-06), f'Rotation about z mismatch: computed {u_y[5]}, expected {theta_z_analytical}'
    P_x = 1000.0
    F_x = np.array([P_x, 0, 0, 0, 0, 0])
    u_x = np.linalg.solve(K_free, F_x)
    u_analytical_x = P_x * L / (E * A)
    assert np.isclose(u_x[0], u_analytical_x, rtol=1e-06), f'X-direction deflection mismatch: computed {u_x[0]}, expected {u_analytical_x}'
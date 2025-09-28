def test_local_stiffness_3D_beam(fcn):
    """
    Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    L = 2.0
    Iy = 1e-06
    Iz = 1e-06
    J = 1e-06
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert K.shape == (12, 12), 'Stiffness matrix should be 12x12.'
    assert np.allclose(K, K.T), 'Stiffness matrix should be symmetric.'
    assert np.linalg.matrix_rank(K) < 12, 'Stiffness matrix should be singular due to rigid body modes.'
    expected_axial_stiffness = E * A / L
    assert np.isclose(K[0, 0], expected_axial_stiffness), 'Axial stiffness does not match expected value.'
    expected_torsional_stiffness = E * J / (2 * (1 + nu) * L)
    assert np.isclose(K[3, 3], expected_torsional_stiffness), 'Torsional stiffness does not match expected value.'
    expected_bending_stiffness_y = 12 * E * Iz / L ** 3
    assert np.isclose(K[1, 1], expected_bending_stiffness_y), 'Bending stiffness about y-axis does not match expected value.'
    expected_bending_stiffness_z = 12 * E * Iy / L ** 3
    assert np.isclose(K[2, 2], expected_bending_stiffness_z), 'Bending stiffness about z-axis does not match expected value.'

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """
    Apply a perpendicular point load in the z direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a perpendicular point load in the y direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a parallel point load in the x direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    L = 2.0
    Iy = 1e-06
    Iz = 1e-06
    J = 1e-06
    K = fcn(E, nu, A, L, Iy, Iz, J)
    Fz = np.zeros(12)
    Fz[8] = -1000
    Fy = np.zeros(12)
    Fy[7] = -1000
    Fx = np.zeros(12)
    Fx[6] = 1000
    displacements_z = np.linalg.solve(K, Fz)
    displacements_y = np.linalg.solve(K, Fy)
    displacements_x = np.linalg.solve(K, Fx)
    deflection_z_analytical = 1000 * L ** 3 / (3 * E * Iz)
    deflection_y_analytical = 1000 * L ** 3 / (3 * E * Iy)
    deflection_x_analytical = 1000 * L / (E * A)
    assert np.isclose(displacements_z[2], deflection_z_analytical, rtol=1e-05), 'Deflection in z does not match analytical solution.'
    assert np.isclose(displacements_y[1], deflection_y_analytical, rtol=1e-05), 'Deflection in y does not match analytical solution.'
    assert np.isclose(displacements_x[0], deflection_x_analytical, rtol=1e-05), 'Deflection in x does not match analytical solution.'
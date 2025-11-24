def test_local_stiffness_3D_beam(fcn):
    """Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    L = 2.0
    Iy = 1e-06
    Iz = 2e-06
    J = 5e-06
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert K.shape == (12, 12), 'Stiffness matrix must be 12x12'
    assert np.allclose(K, K.T), 'Stiffness matrix must be symmetric'
    assert np.linalg.matrix_rank(K) < 12, 'Stiffness matrix should be singular due to rigid body modes'
    expected_axial_stiffness = E * A / L
    assert np.isclose(K[0, 0], expected_axial_stiffness), 'Axial stiffness does not match expected value'
    expected_torsional_stiffness = G * J / L
    assert np.isclose(K[3, 3], expected_torsional_stiffness), 'Torsional stiffness does not match expected value'
    expected_bending_stiffness_y = 12 * E * Iz / L ** 3
    assert np.isclose(K[5, 5], expected_bending_stiffness_y), 'Bending stiffness about y-axis does not match expected value'
    expected_bending_stiffness_z = 12 * E * Iy / L ** 3
    assert np.isclose(K[4, 4], expected_bending_stiffness_z), 'Bending stiffness about z-axis does not match expected value'

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """Apply loads to the tip of a cantilever beam and verify displacements match Euler-Bernoulli beam theory."""
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    L = 2.0
    Iy = 1e-06
    Iz = 2e-06
    J = 5e-06
    K = fcn(E, nu, A, L, Iy, Iz, J)
    Fz = np.zeros(12)
    Fz[8] = -1000
    displacement_z = np.linalg.solve(K, Fz)
    expected_displacement_z = -1000 * L ** 3 / (3 * E * Iz)
    assert np.isclose(displacement_z[8], expected_displacement_z, atol=1e-06), 'Displacement in z direction does not match Euler-Bernoulli theory'
    Fy = np.zeros(12)
    Fy[7] = -1000
    displacement_y = np.linalg.solve(K, Fy)
    expected_displacement_y = -1000 * L ** 3 / (3 * E * Iy)
    assert np.isclose(displacement_y[7], expected_displacement_y, atol=1e-06), 'Displacement in y direction does not match Euler-Bernoulli theory'
    Fx = np.zeros(12)
    Fx[6] = 1000
    displacement_x = np.linalg.solve(K, Fx)
    expected_displacement_x = 1000 * L / (E * A)
    assert np.isclose(displacement_x[6], expected_displacement_x, atol=1e-06), 'Displacement in x direction does not match Euler-Bernoulli theory'
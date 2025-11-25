def test_local_stiffness_3D_beam(fcn):
    """
    Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    L = 2.0
    Iy = 1e-06
    Iz = 2e-06
    J = 0.0001
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert K.shape == (12, 12), 'Stiffness matrix should be 12x12.'
    assert np.allclose(K, K.T), 'Stiffness matrix should be symmetric.'
    assert np.linalg.matrix_rank(K) < 12, 'Stiffness matrix should be singular due to rigid body modes.'
    expected_axial_stiffness = E * A / L
    assert np.isclose(K[0, 0], expected_axial_stiffness), 'Axial stiffness does not match expected value.'
    expected_torsional_stiffness = E * J / (2 * (1 + nu) * L)
    assert np.isclose(K[3, 3], expected_torsional_stiffness), 'Torsional stiffness does not match expected value.'
    expected_bending_stiffness_y = 12 * E * Iz / L ** 3
    expected_bending_stiffness_z = 12 * E * Iy / L ** 3
    assert np.isclose(K[1, 1], expected_bending_stiffness_y), 'Bending stiffness about y does not match expected value.'
    assert np.isclose(K[2, 2], expected_bending_stiffness_z), 'Bending stiffness about z does not match expected value.'

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
    Iz = 2e-06
    J = 0.0001
    K = fcn(E, nu, A, L, Iy, Iz, J)
    Fz = 1000.0
    Fy = 1000.0
    Fx = 1000.0
    delta_z_analytical = Fz * L ** 3 / (3 * E * Iz)
    delta_y_analytical = Fy * L ** 3 / (3 * E * Iy)
    delta_x_analytical = Fx * L / (E * A)
    F = np.zeros(12)
    F[8] = Fz
    F[7] = Fy
    F[6] = Fx
    d = np.linalg.solve(K, F)
    assert np.isclose(d[8], delta_z_analytical, atol=1e-06), 'Displacement in z does not match analytical solution.'
    assert np.isclose(d[7], delta_y_analytical, atol=1e-06), 'Displacement in y does not match analytical solution.'
    assert np.isclose(d[6], delta_x_analytical, atol=1e-06), 'Displacement in x does not match analytical solution.'
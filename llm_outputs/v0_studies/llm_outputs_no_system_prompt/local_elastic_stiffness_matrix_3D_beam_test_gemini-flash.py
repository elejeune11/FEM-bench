def test_local_stiffness_3D_beam(fcn):
    """Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    shape check
    symmetry
    expected singularity due to rigid body modes
    block-level verification of axial, torsion, and bending terms"""
    (E, nu, A, L, Iy, Iz, J) = (1000000000.0, 0.3, 0.1, 1.0, 0.01, 0.01, 0.001)
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T)
    assert np.linalg.matrix_rank(K) == 6
    assert np.allclose(K[0, 0], E * A / L)
    assert np.allclose(K[0, 6], -E * A / L)
    assert np.allclose(K[3, 3], G * J / L)
    assert np.allclose(K[3, 9], -G * J / L)
    assert np.allclose(K[1, 1], 12 * E * Iy / L ** 3)
    assert np.allclose(K[1, 7], -12 * E * Iy / L ** 3)
    assert np.allclose(K[2, 2], 12 * E * Iz / L ** 3)
    assert np.allclose(K[2, 8], -12 * E * Iz / L ** 3)
    G = E / (2 * (1 + nu))

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """Apply a perpendicular point load in the z direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a perpendicular point load in the y direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a parallel point load in the x direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory."""
    (E, nu, A, L, Iy, Iz, J) = (1000000000.0, 0.3, 0.1, 1.0, 0.01, 0.01, 0.001)
    K = fcn(E, nu, A, L, Iy, Iz, J)
    P = np.zeros(12)
    P[8] = 1000
    disp = np.linalg.solve(K, P)
    analytical_z = P[8] * L ** 3 / (3 * E * Iz)
    assert np.allclose(disp[8], analytical_z, rtol=0.001)
    P = np.zeros(12)
    P[7] = 1000
    disp = np.linalg.solve(K, P)
    analytical_y = P[7] * L ** 3 / (3 * E * Iy)
    assert np.allclose(disp[7], analytical_y, rtol=0.001)
    P = np.zeros(12)
    P[6] = 1000
    disp = np.linalg.solve(K, P)
    analytical_x = P[6] * L / (E * A)
    assert np.allclose(disp[6], analytical_x, rtol=0.001)
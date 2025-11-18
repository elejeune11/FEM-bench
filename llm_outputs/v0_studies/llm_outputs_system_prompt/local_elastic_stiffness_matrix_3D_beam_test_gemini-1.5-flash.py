def test_local_stiffness_3D_beam(fcn):
    """Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    shape check
    symmetry
    expected singularity due to rigid body modes
    block-level verification of axial, torsion, and bending terms"""
    (E, nu, A, L, Iy, Iz, J) = (1000000000.0, 0.3, 0.1, 1.0, 0.0001, 0.0001, 1e-05)
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T)
    assert np.allclose(np.linalg.det(K), 0)
    assert np.allclose(K[0, 0], E * A / L)
    assert np.allclose(K[0, 6], -E * A / L)
    assert np.allclose(K[6, 0], -E * A / L)
    assert np.allclose(K[6, 6], E * A / L)
    assert np.allclose(K[1, 1], 12 * E * Iz / L ** 3)
    assert np.allclose(K[1, 7], -12 * E * Iz / L ** 3)
    assert np.allclose(K[1, 5], 6 * E * Iz / L ** 2)
    assert np.allclose(K[1, 11], 6 * E * Iz / L ** 2)
    assert np.allclose(K[2, 2], 12 * E * Iy / L ** 3)
    assert np.allclose(K[2, 8], -12 * E * Iy / L ** 3)
    assert np.allclose(K[2, 4], -6 * E * Iy / L ** 2)
    assert np.allclose(K[2, 10], 6 * E * Iy / L ** 2)
    assert np.allclose(K[3, 3], (1 + nu) * J / L)
    assert np.allclose(K[3, 9], -(1 + nu) * J / L)
    assert np.allclose(K[9, 3], -(1 + nu) * J / L)
    assert np.allclose(K[9, 9], (1 + nu) * J / L)

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """Apply a perpendicular point load in the z direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a perpendicular point load in the y direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a parallel point load in the x direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory."""
    (E, nu, A, L, Iy, Iz, J) = (1000000000.0, 0.3, 0.1, 1.0, 0.0001, 0.0001, 1e-05)
    K = fcn(E, nu, A, L, Iy, Iz, J)
    F = np.zeros(12)
    F[8] = -100
    d = np.linalg.solve(K, F)
    assert np.allclose(d[2], -100 * L ** 3 / (3 * E * Iy))
    F = np.zeros(12)
    F[7] = -100
    d = np.linalg.solve(K, F)
    assert np.allclose(d[1], -100 * L ** 3 / (3 * E * Iz))
    F = np.zeros(12)
    F[6] = -100
    d = np.linalg.solve(K, F)
    assert np.allclose(d[0], -100 * L / (E * A))
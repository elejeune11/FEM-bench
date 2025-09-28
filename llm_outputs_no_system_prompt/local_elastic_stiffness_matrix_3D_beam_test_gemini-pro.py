def test_local_stiffness_3D_beam(fcn):
    """Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    shape check
    symmetry
    expected singularity due to rigid body modes
    block-level verification of axial, torsion, and bending terms"""
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    L = 1.0
    Iy = 1e-06
    Iz = 2e-06
    J = 1e-07
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T)
    assert np.linalg.matrix_rank(K) == 6
    assert np.isclose(K[0, 0], E * A / L)
    assert np.isclose(K[0, 6], -E * A / L)
    assert np.isclose(K[3, 3], J * E / (2 * L * (1 + nu)))
    assert np.isclose(K[3, 9], -J * E / (2 * L * (1 + nu)))
    assert np.isclose(K[1, 1], 12 * E * Iz / L ** 3)
    assert np.isclose(K[1, 5], 6 * E * Iz / L ** 2)
    assert np.isclose(K[1, 7], -12 * E * Iz / L ** 3)
    assert np.isclose(K[1, 11], 6 * E * Iz / L ** 2)
    assert np.isclose(K[2, 2], 12 * E * Iy / L ** 3)
    assert np.isclose(K[2, 4], -6 * E * Iy / L ** 2)
    assert np.isclose(K[2, 8], -12 * E * Iy / L ** 3)
    assert np.isclose(K[2, 10], -6 * E * Iy / L ** 2)

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """Apply a perpendicular point load in the z direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a perpendicular point load in the y direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a parallel point load in the x direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory."""
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    L = 1.0
    Iy = 1e-06
    Iz = 2e-06
    J = 1e-07
    K = fcn(E, nu, A, L, Iy, Iz, J)
    K_reduced = K[6:, 6:]
    Fz = 1000
    F = np.zeros(6)
    F[2] = Fz
    dz = np.linalg.solve(K_reduced, F)[2]
    assert np.isclose(dz, Fz * L ** 3 / (3 * E * Iy))
    Fy = 1000
    F = np.zeros(6)
    F[1] = Fy
    dy = np.linalg.solve(K_reduced, F)[1]
    assert np.isclose(dy, Fy * L ** 3 / (3 * E * Iz))
    Fx = 1000
    F = np.zeros(6)
    F[0] = Fx
    dx = np.linalg.solve(K_reduced, F)[0]
    assert np.isclose(dx, Fx * L / (E * A))
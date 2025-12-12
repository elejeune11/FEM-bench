def test_local_stiffness_3D_beam(fcn):
    """
    Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    """
    E = 200000000000.0
    nu = 0.3
    A = 0.1
    L = 10
    Iy = 0.01
    Iz = 0.02
    J = 0.005
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T)
    assert np.isclose(np.linalg.det(K), 0)
    axial_stiffness = E * A / L
    assert np.isclose(K[0, 0], axial_stiffness)
    assert np.isclose(K[6, 6], axial_stiffness)
    torsion_stiffness = E * J / L
    assert np.isclose(K[3, 3], torsion_stiffness)
    assert np.isclose(K[9, 9], torsion_stiffness)
    bending_stiffness_y = 4 * E * Iy / L ** 3
    bending_stiffness_z = 4 * E * Iz / L ** 3
    assert np.isclose(K[1, 1], bending_stiffness_y)
    assert np.isclose(K[2, 2], bending_stiffness_z)
    assert np.isclose(K[4, 4], bending_stiffness_y)
    assert np.isclose(K[5, 5], bending_stiffness_z)
    assert np.isclose(K[7, 7], bending_stiffness_y)
    assert np.isclose(K[8, 8], bending_stiffness_z)
    assert np.isclose(K[10, 10], bending_stiffness_y)
    assert np.isclose(K[11, 11], bending_stiffness_z)

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """
    Apply a perpendicular point load in the z direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a perpendicular point load in the y direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a parallel point load in the x direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    """
    E = 200000000000.0
    nu = 0.3
    A = 0.1
    L = 10
    Iy = 0.01
    Iz = 0.02
    J = 0.005
    K = fcn(E, nu, A, L, Iy, Iz, J)
    F = np.zeros(12)
    F[8] = 1000
    U = np.linalg.solve(K, F)
    w_tip = U[8]
    assert np.isclose(w_tip, -1000 * L ** 3 / (3 * E * Iz))
    F = np.zeros(12)
    F[7] = 1000
    U = np.linalg.solve(K, F)
    v_tip = U[7]
    assert np.isclose(v_tip, -1000 * L ** 3 / (3 * E * Iy))
    F = np.zeros(12)
    F[6] = 1000
    U = np.linalg.solve(K, F)
    u_tip = U[6]
    assert np.isclose(u_tip, 1000 * L / (E * A))
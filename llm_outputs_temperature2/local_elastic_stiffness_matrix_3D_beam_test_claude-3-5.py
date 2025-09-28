def test_local_stiffness_3D_beam(fcn):
    """
    Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    """
    import numpy as np
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    L = 1.0
    Iy = 0.0001
    Iz = 0.0001
    J = 0.0001
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T)
    eigenvals = np.linalg.eigvals(K)
    assert sum(abs(eigenvals) < 1e-10) == 6
    assert np.isclose(K[0, 0], E * A / L)
    assert np.isclose(K[0, 6], -E * A / L)
    G = E / (2 * (1 + nu))
    assert np.isclose(K[3, 3], G * J / L)
    assert np.isclose(K[3, 9], -G * J / L)
    assert np.isclose(K[1, 1], 12 * E * Iz / L ** 3)
    assert np.isclose(K[1, 5], 6 * E * Iz / L ** 2)
    assert np.isclose(K[5, 5], 4 * E * Iz / L)
    assert np.isclose(K[2, 2], 12 * E * Iy / L ** 3)
    assert np.isclose(K[2, 4], -6 * E * Iy / L ** 2)
    assert np.isclose(K[4, 4], 4 * E * Iy / L)

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """
    Apply point loads to cantilever beam tip and verify displacements match 
    analytical solutions from Euler-Bernoulli beam theory.
    """
    import numpy as np
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    L = 1.0
    Iy = 0.0001
    Iz = 0.0001
    J = 0.0001
    P = 1000.0
    K = fcn(E, nu, A, L, Iy, Iz, J)
    K_red = K[6:, 6:]
    F_z = np.zeros(6)
    F_z[2] = P
    d_z = np.linalg.solve(K_red, F_z)
    analytical_z = P * L ** 3 / (3 * E * Iy)
    assert np.isclose(d_z[2], analytical_z, rtol=1e-10)
    F_y = np.zeros(6)
    F_y[1] = P
    d_y = np.linalg.solve(K_red, F_y)
    analytical_y = P * L ** 3 / (3 * E * Iz)
    assert np.isclose(d_y[1], analytical_y, rtol=1e-10)
    F_x = np.zeros(6)
    F_x[0] = P
    d_x = np.linalg.solve(K_red, F_x)
    analytical_x = P * L / (E * A)
    assert np.isclose(d_x[0], analytical_x, rtol=1e-10)
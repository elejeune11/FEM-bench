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
    near_zero = np.isclose(eigenvals, 0, atol=1e-10)
    assert sum(near_zero) == 6
    EA_L = E * A / L
    assert np.isclose(K[0, 0], EA_L)
    assert np.isclose(K[0, 6], -EA_L)
    GJ_L = E * J / (2 * (1 + nu)) / L
    assert np.isclose(K[3, 3], GJ_L)
    assert np.isclose(K[3, 9], -GJ_L)
    EI_L3 = 12 * E * Iz / L ** 3
    assert np.isclose(K[1, 1], EI_L3)
    assert np.isclose(K[1, 7], -EI_L3)

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """
    Apply point loads to cantilever beam tip and verify displacements match
    analytical solutions from Euler-Bernoulli beam theory for:
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
    K_free = K[6:, 6:]
    P = 1000.0
    F_z = np.zeros(6)
    F_z[2] = P
    d_z = np.linalg.solve(K_free, F_z)
    analytical_z = P * L ** 3 / (3 * E * Iz)
    assert np.isclose(d_z[2], analytical_z, rtol=0.001)
    F_y = np.zeros(6)
    F_y[1] = P
    d_y = np.linalg.solve(K_free, F_y)
    analytical_y = P * L ** 3 / (3 * E * Iy)
    assert np.isclose(d_y[1], analytical_y, rtol=0.001)
    F_x = np.zeros(6)
    F_x[0] = P
    d_x = np.linalg.solve(K_free, F_x)
    analytical_x = P * L / (E * A)
    assert np.isclose(d_x[0], analytical_x, rtol=0.001)
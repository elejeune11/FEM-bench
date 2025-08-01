def test_local_stiffness_3D_beam(fcn):
    """
    Comprehensive test for local_elastic_stiffness_matrix_3D_beam function.
    Verifies matrix shape, symmetry, rigid body modes, and block-level terms.
    """
    import numpy as np
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    L = 2.0
    Iy = 8.33e-06
    Iz = 8.33e-06
    J = 1.41e-05
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T)
    eigenvals = np.linalg.eigvals(K)
    zero_eigenvals = np.sum(np.abs(eigenvals) < 1e-10)
    assert zero_eigenvals == 6
    EA_L = E * A / L
    assert np.isclose(K[0, 0], EA_L)
    assert np.isclose(K[0, 6], -EA_L)
    assert np.isclose(K[6, 6], EA_L)
    GJ_L = E * J / (2 * (1 + nu) * L)
    assert np.isclose(K[3, 3], GJ_L)
    assert np.isclose(K[3, 9], -GJ_L)
    assert np.isclose(K[9, 9], GJ_L)
    EIz_L3 = 12 * E * Iz / L ** 3
    EIz_L2 = 6 * E * Iz / L ** 2
    EIz_L = 4 * E * Iz / L
    assert np.isclose(K[1, 1], EIz_L3)
    assert np.isclose(K[1, 5], EIz_L2)
    assert np.isclose(K[5, 5], EIz_L)
    EIy_L3 = 12 * E * Iy / L ** 3
    EIy_L2 = 6 * E * Iy / L ** 2
    EIy_L = 4 * E * Iy / L
    assert np.isclose(K[2, 2], EIy_L3)
    assert np.isclose(K[2, 4], -EIy_L2)
    assert np.isclose(K[4, 4], EIy_L)
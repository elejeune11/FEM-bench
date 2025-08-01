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
    J = 5e-07
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T)
    assert np.linalg.matrix_rank(K) == 6
    EA_L = E * A / L
    GJ_L = E * J / L
    EIy_L3 = E * Iy / L ** 3
    EIz_L3 = E * Iz / L ** 3
    assert np.allclose(K[0, 0], EA_L)
    assert np.allclose(K[0, 6], -EA_L)
    assert np.allclose(K[6, 6], EA_L)
    assert np.allclose(K[3, 3], GJ_L)
    assert np.allclose(K[3, 9], -GJ_L)
    assert np.allclose(K[9, 9], GJ_L)
    assert np.allclose(K[1, 1], 12 * EIy_L3)
    assert np.allclose(K[1, 5], 6 * L * EIy_L3)
    assert np.allclose(K[1, 7], -12 * EIy_L3)
    assert np.allclose(K[1, 11], 6 * L * EIy_L3)
    assert np.allclose(K[5, 5], 4 * L ** 2 * EIy_L3)
    assert np.allclose(K[5, 7], -6 * L * EIy_L3)
    assert np.allclose(K[5, 11], 2 * L ** 2 * EIy_L3)
    assert np.allclose(K[7, 7], 12 * EIy_L3)
    assert np.allclose(K[7, 11], -6 * L * EIy_L3)
    assert np.allclose(K[11, 11], 4 * L ** 2 * EIy_L3)
    assert np.allclose(K[2, 2], 12 * EIz_L3)
    assert np.allclose(K[2, 4], -6 * L * EIz_L3)
    assert np.allclose(K[2, 8], -12 * EIz_L3)
    assert np.allclose(K[2, 10], -6 * L * EIz_L3)
    assert np.allclose(K[4, 4], 4 * L ** 2 * EIz_L3)
    assert np.allclose(K[4, 8], 6 * L * EIz_L3)
    assert np.allclose(K[4, 10], 2 * L ** 2 * EIz_L3)
    assert np.allclose(K[8, 8], 12 * EIz_L3)
    assert np.allclose(K[8, 10], 6 * L * EIz_L3)
    assert np.allclose(K[10, 10], 4 * L ** 2 * EIz_L3)
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
    Iy = 1e-05
    Iz = 1e-05
    J = 1e-06
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T)
    assert np.linalg.matrix_rank(K) == 6
    EA_over_L = E * A / L
    assert np.isclose(K[0, 0], EA_over_L)
    assert np.isclose(K[6, 6], EA_over_L)
    assert np.isclose(K[0, 6], -EA_over_L)
    assert np.isclose(K[6, 0], -EA_over_L)
    GJ_over_L = E * J / (2 * (1 + nu) * L)
    assert np.isclose(K[3, 3], GJ_over_L)
    assert np.isclose(K[9, 9], GJ_over_L)
    assert np.isclose(K[3, 9], -GJ_over_L)
    assert np.isclose(K[9, 3], -GJ_over_L)
    EIy_over_L3 = E * Iy / L ** 3
    EIy_over_L = E * Iy / L
    assert np.isclose(K[1, 1], 12 * EIy_over_L3)
    assert np.isclose(K[7, 7], 12 * EIy_over_L3)
    assert np.isclose(K[1, 7], 6 * EIy_over_L3 * L)
    assert np.isclose(K[7, 1], 6 * EIy_over_L3 * L)
    EIz_over_L3 = E * Iz / L ** 3
    EIz_over_L = E * Iz / L
    assert np.isclose(K[2, 2], 12 * EIz_over_L3)
    assert np.isclose(K[8, 8], 12 * EIz_over_L3)
    assert np.isclose(K[2, 8], 6 * EIz_over_L3 * L)
    assert np.isclose(K[8, 2], 6 * EIz_over_L3 * L)
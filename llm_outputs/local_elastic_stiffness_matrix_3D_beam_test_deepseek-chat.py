def test_local_stiffness_3D_beam(fcn):
    """
    Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    """
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    L = 2.0
    Iy = 1e-05
    Iz = 2e-05
    J = 1e-06
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T)
    assert np.linalg.matrix_rank(K) == 6
    axial_stiffness = E * A / L
    torsion_stiffness = E * J / (2 * (1 + nu) * L)
    bending_y_stiffness = 12 * E * Iy / L ** 3
    bending_z_stiffness = 12 * E * Iz / L ** 3
    assert np.isclose(K[0, 0], axial_stiffness)
    assert np.isclose(K[0, 6], -axial_stiffness)
    assert np.isclose(K[6, 6], axial_stiffness)
    assert np.isclose(K[3, 3], torsion_stiffness)
    assert np.isclose(K[3, 9], -torsion_stiffness)
    assert np.isclose(K[9, 9], torsion_stiffness)
    assert np.isclose(K[1, 1], bending_y_stiffness)
    assert np.isclose(K[1, 5], 6 * E * Iy / L ** 2)
    assert np.isclose(K[1, 7], -bending_y_stiffness)
    assert np.isclose(K[5, 5], 4 * E * Iy / L)
    assert np.isclose(K[2, 2], bending_z_stiffness)
    assert np.isclose(K[2, 4], -6 * E * Iz / L ** 2)
    assert np.isclose(K[2, 8], -bending_z_stiffness)
    assert np.isclose(K[4, 4], 4 * E * Iz / L)
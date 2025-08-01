def test_local_stiffness_3D_beam(fcn):
    """
    Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    L = 2.0
    Iy = 1e-06
    Iz = 2e-06
    J = 1e-06
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert K.shape == (12, 12), 'Stiffness matrix should be 12x12'
    assert np.allclose(K, K.T), 'Stiffness matrix should be symmetric'
    assert np.isclose(np.linalg.det(K), 0), 'Stiffness matrix should be singular for a free-free beam'
    expected_axial_stiffness = E * A / L
    assert np.isclose(K[0, 0], expected_axial_stiffness), 'Axial stiffness term is incorrect'
    assert np.isclose(K[6, 6], expected_axial_stiffness), 'Axial stiffness term is incorrect'
    assert np.isclose(K[0, 6], -expected_axial_stiffness), 'Axial stiffness coupling term is incorrect'
    assert np.isclose(K[6, 0], -expected_axial_stiffness), 'Axial stiffness coupling term is incorrect'
    expected_torsional_stiffness = E * J / (2 * (1 + nu) * L)
    assert np.isclose(K[3, 3], expected_torsional_stiffness), 'Torsional stiffness term is incorrect'
    assert np.isclose(K[9, 9], expected_torsional_stiffness), 'Torsional stiffness term is incorrect'
    assert np.isclose(K[3, 9], -expected_torsional_stiffness), 'Torsional stiffness coupling term is incorrect'
    assert np.isclose(K[9, 3], -expected_torsional_stiffness), 'Torsional stiffness coupling term is incorrect'
    expected_bending_stiffness_y = 12 * E * Iz / L ** 3
    assert np.isclose(K[1, 1], expected_bending_stiffness_y), 'Bending stiffness term about y-axis is incorrect'
    assert np.isclose(K[7, 7], expected_bending_stiffness_y), 'Bending stiffness term about y-axis is incorrect'
    assert np.isclose(K[1, 7], -expected_bending_stiffness_y), 'Bending stiffness coupling term about y-axis is incorrect'
    assert np.isclose(K[7, 1], -expected_bending_stiffness_y), 'Bending stiffness coupling term about y-axis is incorrect'
    expected_bending_stiffness_z = 12 * E * Iy / L ** 3
    assert np.isclose(K[2, 2], expected_bending_stiffness_z), 'Bending stiffness term about z-axis is incorrect'
    assert np.isclose(K[8, 8], expected_bending_stiffness_z), 'Bending stiffness term about z-axis is incorrect'
    assert np.isclose(K[2, 8], -expected_bending_stiffness_z), 'Bending stiffness coupling term about z-axis is incorrect'
    assert np.isclose(K[8, 2], -expected_bending_stiffness_z), 'Bending stiffness coupling term about z-axis is incorrect'
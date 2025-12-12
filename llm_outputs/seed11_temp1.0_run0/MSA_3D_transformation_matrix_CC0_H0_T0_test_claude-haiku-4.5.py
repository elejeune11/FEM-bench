def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    """
    x_axis_ref = np.array([0.0, 0.0, 1.0])
    gamma_x = fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, x_axis_ref)
    dcm_x = gamma_x[:3, :3]
    expected_local_x_x = np.array([1.0, 0.0, 0.0])
    expected_local_y_x = np.array([0.0, 1.0, 0.0])
    expected_local_z_x = np.array([0.0, 0.0, 1.0])
    assert np.allclose(dcm_x[0, :], expected_local_x_x), 'X-axis beam local_x incorrect'
    assert np.allclose(dcm_x[1, :], expected_local_y_x), 'X-axis beam local_y incorrect'
    assert np.allclose(dcm_x[2, :], expected_local_z_x), 'X-axis beam local_z incorrect'
    y_axis_ref = np.array([0.0, 0.0, 1.0])
    gamma_y = fcn(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, y_axis_ref)
    dcm_y = gamma_y[:3, :3]
    expected_local_x_y = np.array([0.0, 1.0, 0.0])
    expected_local_y_y = np.array([-1.0, 0.0, 0.0])
    expected_local_z_y = np.array([0.0, 0.0, 1.0])
    assert np.allclose(dcm_y[0, :], expected_local_x_y), 'Y-axis beam local_x incorrect'
    assert np.allclose(dcm_y[1, :], expected_local_y_y), 'Y-axis beam local_y incorrect'
    assert np.allclose(dcm_y[2, :], expected_local_z_y), 'Y-axis beam local_z incorrect'
    z_axis_ref = np.array([0.0, 1.0, 0.0])
    gamma_z = fcn(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, z_axis_ref)
    dcm_z = gamma_z[:3, :3]
    expected_local_x_z = np.array([0.0, 0.0, 1.0])
    expected_local_y_z = np.array([0.0, 1.0, 0.0])
    expected_local_z_z = np.array([-1.0, 0.0, 0.0])
    assert np.allclose(dcm_z[0, :], expected_local_x_z), 'Z-axis beam local_x incorrect'
    assert np.allclose(dcm_z[1, :], expected_local_y_z), 'Z-axis beam local_y incorrect'
    assert np.allclose(dcm_z[2, :], expected_local_z_z), 'Z-axis beam local_z incorrect'

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the transformation is correct.
    """
    ref_vector = np.array([0.0, 0.0, 1.0])
    gamma = fcn(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, ref_vector)
    assert gamma.shape == (12, 12), 'Transformation matrix should be 12x12'
    dcm = gamma[:3, :3]
    assert np.allclose(gamma[3:6, 3:6], dcm), 'Second block should match first block'
    assert np.allclose(gamma[6:9, 6:9], dcm), 'Third block should match first block'
    assert np.allclose(gamma[9:12, 9:12], dcm), 'Fourth block should match first block'
    assert np.allclose(dcm @ dcm.T, np.eye(3)), 'Direction cosine matrix should be orthogonal'
    assert np.allclose(np.linalg.det(dcm), 1.0), 'Direction cosine matrix determinant should be 1'
    ref_vector_2 = np.array([1.0, 0.0, 0.0])
    gamma_2 = fcn(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, ref_vector_2)
    dcm_2 = gamma_2[:3, :3]
    assert np.allclose(dcm_2 @ dcm_2.T, np.eye(3)), 'Second DCM should be orthogonal'
    assert np.allclose(np.linalg.det(dcm_2), 1.0), 'Second DCM determinant should be 1'
    beam_dir = np.array([3.0, 4.0, 0.0])
    beam_dir_normalized = beam_dir / np.linalg.norm(beam_dir)
    ref_vector_3 = np.array([0.0, 0.0, 1.0])
    gamma_3 = fcn(0.0, 0.0, 0.0, beam_dir[0], beam_dir[1], beam_dir[2], ref_vector_3)
    dcm_3 = gamma_3[:3, :3]
    assert np.allclose(dcm_3[0, :], beam_dir_normalized), 'Local x-axis should align with beam direction'

def test_beam_transformation_matrix_error_messages(fcn):
    """Test that the function raises appropriate ValueError exceptions for invalid reference vectors.
    Verifies proper error handling for invalid reference vectors and zero-length beams.
    """
    non_unit_ref = np.array([1.0, 1.0, 0.0])
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, non_unit_ref)
    parallel_ref = np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, parallel_ref)
    parallel_ref_z = np.array([0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, parallel_ref_z)
    valid_ref = np.array([0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        fcn(1.0, 2.0, 3.0, 1.0, 2.0, 3.0, valid_ref)
    wrong_shape_ref = np.array([0.0, 0.0, 1.0, 0.0])
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, wrong_shape_ref)
    wrong_shape_ref_2d = np.array([[0.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, wrong_shape_ref_2d)
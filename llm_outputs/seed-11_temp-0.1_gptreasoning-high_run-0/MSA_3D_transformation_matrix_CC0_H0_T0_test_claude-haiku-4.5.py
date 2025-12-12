def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    """
    ref_z = np.array([0.0, 0.0, 1.0])
    gamma_x = fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, ref_z)
    dc_x = gamma_x[:3, :3]
    expected_local_x = np.array([1.0, 0.0, 0.0])
    assert np.allclose(dc_x[0, :], expected_local_x), 'X-axis beam local x-axis incorrect'
    gamma_y = fcn(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, ref_z)
    dc_y = gamma_y[:3, :3]
    expected_local_x_y = np.array([0.0, 1.0, 0.0])
    assert np.allclose(dc_y[0, :], expected_local_x_y), 'Y-axis beam local x-axis incorrect'
    ref_y = np.array([0.0, 1.0, 0.0])
    gamma_z = fcn(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, ref_y)
    dc_z = gamma_z[:3, :3]
    expected_local_x_z = np.array([0.0, 0.0, 1.0])
    assert np.allclose(dc_z[0, :], expected_local_x_z), 'Z-axis beam local x-axis incorrect'
    for dc in [dc_x, dc_y, dc_z]:
        assert np.allclose(dc @ dc.T, np.eye(3)), 'Direction cosine matrix not orthogonal'
        assert np.allclose(np.linalg.det(dc), 1.0), 'Direction cosine matrix determinant not 1'

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the transformation is correct.
    """
    ref_vec = np.array([0.0, 0.0, 1.0])
    gamma = fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, ref_vec)
    assert gamma.shape == (12, 12), 'Transformation matrix should be 12x12'
    dc_block = gamma[:3, :3]
    for i in range(4):
        start_idx = i * 3
        end_idx = start_idx + 3
        block = gamma[start_idx:end_idx, start_idx:end_idx]
        assert np.allclose(block, dc_block), f'Block {i} does not match direction cosine matrix'
    for i in range(4):
        for j in range(4):
            if i != j:
                block = gamma[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3]
                assert np.allclose(block, np.zeros((3, 3))), f'Off-diagonal block ({i},{j}) should be zero'
    assert np.allclose(gamma.T @ gamma, np.eye(12)), 'Transformation matrix should be orthogonal'
    det_gamma = np.linalg.det(gamma)
    assert np.allclose(det_gamma, 1.0), 'Transformation matrix determinant should be 1'
    gamma2 = fcn(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, ref_vec)
    assert gamma2.shape == (12, 12), 'Transformation matrix for diagonal beam should be 12x12'
    assert np.allclose(gamma2.T @ gamma2, np.eye(12)), 'Diagonal beam transformation should be orthogonal'
    dc = gamma[:3, :3]
    assert np.allclose(np.linalg.norm(dc[0, :]), 1.0), 'First row of DC matrix should be unit vector'
    assert np.allclose(np.linalg.norm(dc[1, :]), 1.0), 'Second row of DC matrix should be unit vector'
    assert np.allclose(np.linalg.norm(dc[2, :]), 1.0), 'Third row of DC matrix should be unit vector'
    assert np.allclose(np.dot(dc[0, :], dc[1, :]), 0.0), 'Rows 1 and 2 should be orthogonal'
    assert np.allclose(np.dot(dc[1, :], dc[2, :]), 0.0), 'Rows 2 and 3 should be orthogonal'
    assert np.allclose(np.dot(dc[0, :], dc[2, :]), 0.0), 'Rows 1 and 3 should be orthogonal'

def test_beam_transformation_matrix_error_messages(fcn):
    """Test that the function raises appropriate ValueError exceptions for invalid inputs.
    Verifies proper error handling for:
    1. Non-unit reference vector
    2. Reference vector parallel to beam axis
    3. Zero-length beam
    """
    non_unit_ref = np.array([1.0, 1.0, 0.0])
    with pytest.raises(ValueError, match='unit vector'):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, non_unit_ref)
    parallel_ref = np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match='parallel'):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, parallel_ref)
    parallel_ref_z = np.array([0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match='parallel'):
        fcn(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, parallel_ref_z)
    ref_vec = np.array([0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match='zero length'):
        fcn(1.0, 2.0, 3.0, 1.0, 2.0, 3.0, ref_vec)
    wrong_shape_ref = np.array([0.0, 0.0])
    with pytest.raises(ValueError, match='shape'):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, wrong_shape_ref)
    wrong_shape_ref_2d = np.array([[0.0], [0.0], [1.0]])
    with pytest.raises(ValueError, match='shape'):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, wrong_shape_ref_2d)
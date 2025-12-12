def test_cardinal_axis_alignment(fcn):
    gamma = fcn(0, 0, 0, 1, 0, 0, np.array([0, 0, 1]))
    dir_cos = gamma[:3, :3]
    assert np.allclose(dir_cos, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])), 'X-axis beam should have identity direction cosine matrix'
    gamma = fcn(0, 0, 0, 0, 1, 0, np.array([0, 0, 1]))
    dir_cos = gamma[:3, :3]
    assert np.allclose(dir_cos, np.array([[0, 1, 0], [0, 0, 1], [-1, 0, 0]])), 'Y-axis beam should have correct direction cosine matrix'
    gamma = fcn(0, 0, 0, 0, 0, 1, None)
    dir_cos = gamma[:3, :3]
    assert np.allclose(dir_cos, np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])), 'Z-axis beam should use global y-axis as reference and have correct orientation'

def test_transformation_matrix_properties(fcn):
    gamma = fcn(0, 0, 0, 1, 1, 1, np.array([1, 0, 0]))
    dir_cos = gamma[:3, :3]
    local_x = np.array([1, 1, 1]) / np.sqrt(3)
    ref_vec = np.array([1, 0, 0])
    local_y = np.cross(ref_vec, local_x)
    local_y = local_y / np.linalg.norm(local_y)
    local_z = np.cross(local_x, local_y)
    expected_dir_cos = np.column_stack([local_x, local_y, local_z])
    assert np.allclose(dir_cos, expected_dir_cos), 'Direction cosine matrix does not match expected for diagonal beam'
    assert np.allclose(gamma.T @ gamma, np.eye(12)), 'Transformation matrix must be orthogonal'
    for i in range(4):
        block = gamma[3 * i:3 * i + 3, 3 * i:3 * i + 3]
        assert np.allclose(block, dir_cos), f'Block {i} does not match first direction cosine matrix'

def test_beam_transformation_matrix_error_messages(fcn):
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, np.array([2, 0, 0]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0, 0]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 0, 0, 0, np.array([0, 0, 1]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0, 0, 0]))
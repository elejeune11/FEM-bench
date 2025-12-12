def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    Gamma_x = fcn(0, 0, 0, 1, 0, 0, None)
    R_x = Gamma_x[0:3, 0:3]
    expected_R_x = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert np.allclose(R_x, expected_R_x, atol=1e-10), f'X-axis beam failed: got {R_x}'
    Gamma_y = fcn(0, 0, 0, 0, 1, 0, None)
    R_y = Gamma_y[0:3, 0:3]
    expected_R_y = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    assert np.allclose(R_y, expected_R_y, atol=1e-10), f'Y-axis beam failed: got {R_y}'
    Gamma_z = fcn(0, 0, 0, 0, 0, 1, None)
    R_z = Gamma_z[0:3, 0:3]
    expected_R_z = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    assert np.allclose(R_z, expected_R_z, atol=1e-10), f'Z-axis beam failed: got {R_z}'

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct.
    """
    test_cases = [(0, 0, 0, 1, 0, 0, None), (0, 0, 0, 0, 1, 0, None), (0, 0, 0, 0, 0, 1, None), (0, 0, 0, 1, 1, 0, None), (0, 0, 0, 1, 1, 1, None), (1, 2, 3, 4, 5, 6, None), (0, 0, 0, 1, 0, 0, np.array([0, 1, 0]))]
    for (x1, y1, z1, x2, y2, z2, ref_vec) in test_cases:
        Gamma = fcn(x1, y1, z1, x2, y2, z2, ref_vec)
        assert Gamma.shape == (12, 12), f'Shape should be (12, 12), got {Gamma.shape}'
        product = Gamma.T @ Gamma
        assert np.allclose(product, np.eye(12), atol=1e-10), 'Gamma should be orthogonal'
        product2 = Gamma @ Gamma.T
        assert np.allclose(product2, np.eye(12), atol=1e-10), 'Gamma @ Gamma.T should be identity'
        det = np.linalg.det(Gamma)
        assert np.isclose(det, 1.0, atol=1e-10), f'Determinant should be 1, got {det}'
        R = Gamma[0:3, 0:3]
        for i in range(4):
            block = Gamma[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3]
            assert np.allclose(block, R, atol=1e-10), f'Block {i} should equal first block'
        for i in range(4):
            for j in range(4):
                if i != j:
                    block = Gamma[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3]
                    assert np.allclose(block, np.zeros((3, 3)), atol=1e-10), f'Off-diagonal block ({i},{j}) should be zero'
        R_det = np.linalg.det(R)
        assert np.isclose(R_det, 1.0, atol=1e-10), f'3x3 block determinant should be 1, got {R_det}'
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10), '3x3 block should be orthogonal'

def test_beam_transformation_matrix_error_messages(fcn):
    """Test that the function raises appropriate ValueError exceptions for invalid reference vectors.
    Verifies proper error handling for two invalid reference vector conditions:
    1. Non-unit vector: When the reference vector's magnitude is not equal to 1.0
    2. Parallel reference vector: When the reference vector is parallel to the beam axis
    Also veriefies proper error handling for:
    3. Zero-length beam: start and end coordinates are identical
    All conditions should raise ValueError
    """
    non_unit_vector = np.array([1, 1, 0])
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, non_unit_vector)
    non_unit_vector2 = np.array([0.5, 0, 0])
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 0, 1, 0, non_unit_vector2)
    parallel_vector_x = np.array([1, 0, 0])
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, parallel_vector_x)
    parallel_vector_y = np.array([0, 1, 0])
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 0, 1, 0, parallel_vector_y)
    parallel_vector_z = np.array([0, 0, 1])
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 0, 0, 1, parallel_vector_z)
    diag_vec = np.array([1, 1, 1]) / np.sqrt(3)
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 1, 1, diag_vec)
    anti_parallel = np.array([-1, 0, 0])
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, anti_parallel)
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 0, 0, 0, None)
    with pytest.raises(ValueError):
        fcn(1, 2, 3, 1, 2, 3, None)
    with pytest.raises(ValueError):
        fcn(5.5, -3.2, 7.1, 5.5, -3.2, 7.1, np.array([0, 0, 1]))
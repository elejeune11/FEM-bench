def test_cardinal_axis_alignment(fcn):
    """
    Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    Gamma_x = fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.array([0.0, 0.0, 1.0]))
    R_x = Gamma_x[:3, :3]
    expected_rows_x = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    expected_cols_x = expected_rows_x.T
    assert np.allclose(R_x, expected_rows_x, atol=1e-12) or np.allclose(R_x, expected_cols_x, atol=1e-12)
    Gamma_y = fcn(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, np.array([0.0, 0.0, 1.0]))
    R_y = Gamma_y[:3, :3]
    expected_rows_y = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    expected_cols_y = expected_rows_y.T
    assert np.allclose(R_y, expected_rows_y, atol=1e-12) or np.allclose(R_y, expected_cols_y, atol=1e-12)
    Gamma_z = fcn(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, None)
    R_z = Gamma_z[:3, :3]
    expected_rows_z = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    expected_cols_z = expected_rows_z.T
    assert np.allclose(R_z, expected_rows_z, atol=1e-12) or np.allclose(R_z, expected_cols_z, atol=1e-12)

def test_transformation_matrix_properties(fcn):
    """
    Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Verifies:
    """

    def check_case(p1, p2, ref_vec):
        Gamma = fcn(*p1, *p2, ref_vec)
        R = Gamma[:3, :3]
        x_dir = np.array(p2) - np.array(p1)
        x_dir = x_dir / np.linalg.norm(x_dir)
        if ref_vec is None:
            global_z = np.array([0.0, 0.0, 1.0])
            global_y = np.array([0.0, 1.0, 0.0])
            if np.allclose(np.abs(x_dir), np.array([0.0, 0.0, 1.0]), atol=1e-12):
                ref = global_y
            else:
                ref = global_z
        else:
            ref = np.array(ref_vec, dtype=float)
            ref = ref / np.linalg.norm(ref)
        y_dir = np.cross(ref, x_dir)
        y_norm = np.linalg.norm(y_dir)
        assert y_norm > 1e-12
        y_dir = y_dir / y_norm
        z_dir = np.cross(x_dir, y_dir)
        z_dir = z_dir / np.linalg.norm(z_dir)
        expected_rows = np.vstack([x_dir, y_dir, z_dir])
        expected_cols = np.column_stack([x_dir, y_dir, z_dir])
        assert np.allclose(R, expected_rows, atol=1e-12) or np.allclose(R, expected_cols, atol=1e-12)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)
        assert Gamma.shape == (12, 12)
        for i in range(4):
            block = Gamma[3 * i:3 * (i + 1), 3 * i:3 * (i + 1)]
            assert np.allclose(block, R, atol=1e-12)
        zeros = np.zeros((3, 3))
        for i in range(4):
            for j in range(4):
                if i != j:
                    block = Gamma[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)]
                    assert np.allclose(block, zeros, atol=1e-12)
        I12 = np.eye(12)
        assert np.allclose(Gamma @ Gamma.T, I12, atol=1e-12)
        assert np.allclose(Gamma.T @ Gamma, I12, atol=1e-12)
        rng = np.random.default_rng(42)
        v = rng.standard_normal(12)
        assert np.isclose(np.linalg.norm(Gamma @ v), np.linalg.norm(v), rtol=1e-12, atol=1e-12)
        assert np.isclose(np.linalg.norm(Gamma.T @ v), np.linalg.norm(v), rtol=1e-12, atol=1e-12)
    check_case((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), np.array([0.0, 0.0, 1.0]))
    check_case((0.0, 0.0, 0.0), (0.0, 1.0, 1.0), np.array([1.0, 0.0, 0.0]))

def test_beam_transformation_matrix_error_messages(fcn):
    """
    Test that the function raises appropriate ValueError exceptions for invalid reference vectors
    and configurations:
    1. Non-unit vector: reference vector magnitude not equal to 1.0
    2. Parallel reference vector: reference vector parallel to the beam axis
    3. Zero-length beam: start and end coordinates are identical
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.array([0.0, 0.0, 2.0]))
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.array([0.0, 0.0, 1.0]))
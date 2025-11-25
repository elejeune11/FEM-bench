def test_cardinal_axis_alignment(fcn):
    """
    Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """

    def compute_expected_R(p1, p2, ref):
        x_vec = np.array(p2) - np.array(p1)
        x = x_vec / np.linalg.norm(x_vec)
        y = np.cross(ref, x)
        y = y / np.linalg.norm(y)
        z = np.cross(x, y)
        z = z / np.linalg.norm(z)
        return np.column_stack((x, y, z))
    ez = np.array([0.0, 0.0, 1.0])
    ey = np.array([0.0, 1.0, 0.0])
    Gamma_x = fcn(0.0, 0.0, 0.0, 5.0, 0.0, 0.0, ez)
    R_x = Gamma_x[:3, :3]
    R_x_expected = compute_expected_R([0, 0, 0], [5, 0, 0], ez)
    assert Gamma_x.shape == (12, 12)
    assert np.allclose(R_x, R_x_expected, atol=1e-12)
    Gamma_y = fcn(0.0, 0.0, 0.0, 0.0, 7.0, 0.0, ez)
    R_y = Gamma_y[:3, :3]
    R_y_expected = compute_expected_R([0, 0, 0], [0, 7, 0], ez)
    assert Gamma_y.shape == (12, 12)
    assert np.allclose(R_y, R_y_expected, atol=1e-12)
    Gamma_z = fcn(0.0, 0.0, 0.0, 0.0, 0.0, 9.0, ey)
    R_z = Gamma_z[:3, :3]
    R_z_expected = compute_expected_R([0, 0, 0], [0, 0, 9], ey)
    assert Gamma_z.shape == (12, 12)
    assert np.allclose(R_z, R_z_expected, atol=1e-12)

def test_transformation_matrix_properties(fcn):
    """
    Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct.
    """

    def unit(v):
        v = np.asarray(v, dtype=float)
        return v / np.linalg.norm(v)
    ez = np.array([0.0, 0.0, 1.0])
    ey = np.array([0.0, 1.0, 0.0])
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 1.0, 0.0])
    Gamma = fcn(*p1, *p2, ez)
    R = Gamma[:3, :3]
    x = unit(p2 - p1)
    y = unit(np.cross(ez, x))
    z = unit(np.cross(x, y))
    R_expected = np.column_stack((x, y, z))
    assert Gamma.shape == (12, 12)
    assert np.allclose(R, R_expected, atol=1e-12)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)
    blocks = [Gamma[i:i + 3, i:i + 3] for i in (0, 3, 6, 9)]
    for B in blocks:
        assert np.allclose(B, R, atol=1e-12)
    assert np.allclose(Gamma.T @ Gamma, np.eye(12), atol=1e-12)
    p1 = np.array([2.0, -1.0, 0.5])
    p2 = np.array([3.0, 1.0, 2.5])
    Gamma_a = fcn(*p1, *p2, ez)
    R_a = Gamma_a[:3, :3]
    Gamma_a_default = fcn(*p1, *p2, None)
    assert np.allclose(Gamma_a, Gamma_a_default, atol=1e-12)
    Gamma_b = fcn(*p2, *p1, ez)
    R_b = Gamma_b[:3, :3]
    assert np.allclose(R_b[:, 0], -R_a[:, 0], atol=1e-12)
    assert np.allclose(R_b[:, 1], -R_a[:, 1], atol=1e-12)
    assert np.allclose(R_b[:, 2], R_a[:, 2], atol=1e-12)
    p1v = np.array([0.0, 0.0, 0.0])
    p2v = np.array([0.0, 0.0, 5.0])
    Gamma_v_yref = fcn(*p1v, *p2v, ey)
    Gamma_v_default = fcn(*p1v, *p2v, None)
    assert np.allclose(Gamma_v_yref, Gamma_v_default, atol=1e-12)

def test_beam_transformation_matrix_error_messages(fcn):
    """
    Test that the function raises appropriate ValueError exceptions for invalid reference vectors.
    Verifies proper error handling for two invalid reference vector conditions:
    1. Non-unit vector: When the reference vector's magnitude is not equal to 1.0
    2. Parallel reference vector: When the reference vector is parallel to the beam axis
    Also veriefies proper error handling for:
    3. Zero-length beam: start and end coordinates are identical
    All conditions should raise ValueError
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 2.0, 0.0, 0.0, np.array([0.0, 0.0, 2.0]))
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 2.0, 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        fcn(1.0, -1.0, 0.5, 1.0, -1.0, 0.5, np.array([0.0, 0.0, 1.0]))
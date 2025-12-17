def test_cardinal_axis_alignment(fcn):
    """
    Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the first 3x3 direction cosine block has correct orientation
    for X-, Y-, and Z-axis aligned beams according to the rules:
    Test cases:
    """

    def expected_dcm(p1, p2, ref):
        v = p2 - p1
        xhat = v / np.linalg.norm(v)
        if ref is None:
            if np.allclose(np.abs(xhat), np.array([0.0, 0.0, 1.0])):
                ref_used = np.array([0.0, 1.0, 0.0])
            else:
                ref_used = np.array([0.0, 0.0, 1.0])
        else:
            ref_used = ref
        ytemp = np.cross(ref_used, xhat)
        yhat = ytemp / np.linalg.norm(ytemp)
        zhat = np.cross(xhat, yhat)
        return np.vstack([xhat, yhat, zhat])
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([2.0, 0.0, 0.0])
    ref = np.array([0.0, 0.0, 1.0])
    Gamma_x = fcn(*p1, *p2, reference_vector=ref)
    R_x_expected = expected_dcm(p1, p2, ref)
    R_x = Gamma_x[:3, :3]
    assert np.allclose(R_x, R_x_expected, atol=1e-12)
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([0.0, 4.0, 0.0])
    ref = np.array([0.0, 0.0, 1.0])
    Gamma_y = fcn(*p1, *p2, reference_vector=ref)
    R_y_expected = expected_dcm(p1, p2, ref)
    R_y = Gamma_y[:3, :3]
    assert np.allclose(R_y, R_y_expected, atol=1e-12)
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([0.0, 0.0, 3.0])
    Gamma_z = fcn(*p1, *p2, reference_vector=None)
    R_z_expected = expected_dcm(p1, p2, None)
    R_z = Gamma_z[:3, :3]
    assert np.allclose(R_z, R_z_expected, atol=1e-12)
    for R in (R_x, R_y, R_z):
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.allclose(np.cross(R[0], R[1]), R[2], atol=1e-12)

def test_transformation_matrix_properties(fcn):
    """
    Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    """

    def expected_dcm(p1, p2, ref):
        v = p2 - p1
        xhat = v / np.linalg.norm(v)
        if ref is None:
            if np.allclose(np.abs(xhat), np.array([0.0, 0.0, 1.0])):
                ref_used = np.array([0.0, 1.0, 0.0])
            else:
                ref_used = np.array([0.0, 0.0, 1.0])
        else:
            ref_used = ref
        ytemp = np.cross(ref_used, xhat)
        yhat = ytemp / np.linalg.norm(ytemp)
        zhat = np.cross(xhat, yhat)
        return np.vstack([xhat, yhat, zhat])
    p1_a = np.array([1.0, -2.0, 0.5])
    p2_a = np.array([3.0, 1.0, 0.5])
    Gamma_a = fcn(*p1_a, *p2_a, reference_vector=None)
    assert Gamma_a.shape == (12, 12)
    R_a = Gamma_a[:3, :3]
    R_a_expected = expected_dcm(p1_a, p2_a, None)
    assert np.allclose(R_a, R_a_expected, atol=1e-12)
    Gamma_a_expected = np.zeros((12, 12))
    for i in range(4):
        Gamma_a_expected[3 * i:3 * (i + 1), 3 * i:3 * (i + 1)] = R_a_expected
    assert np.allclose(Gamma_a, Gamma_a_expected, atol=1e-12)
    I12 = np.eye(12)
    assert np.allclose(Gamma_a @ Gamma_a.T, I12, atol=1e-12)
    assert np.allclose(Gamma_a.T @ Gamma_a, I12, atol=1e-12)
    assert np.isclose(np.linalg.det(R_a), 1.0, atol=1e-12)
    assert np.allclose(np.cross(R_a[0], R_a[1]), R_a[2], atol=1e-12)
    rng = np.random.default_rng(42)
    d_global = rng.normal(size=12)
    d_local = Gamma_a @ d_global
    d_global_roundtrip = Gamma_a.T @ d_local
    assert np.allclose(d_global_roundtrip, d_global, atol=1e-12)
    p1_b = np.array([0.3, -0.5, 1.2])
    p2_b = np.array([2.0, 0.7, 3.1])
    ref_b = np.array([0.2, 0.5, 0.84])
    ref_b = ref_b / np.linalg.norm(ref_b)
    Gamma_b = fcn(*p1_b, *p2_b, reference_vector=ref_b)
    R_b = Gamma_b[:3, :3]
    R_b_expected = expected_dcm(p1_b, p2_b, ref_b)
    assert np.allclose(R_b, R_b_expected, atol=1e-12)
    Gamma_b_expected = np.zeros((12, 12))
    for i in range(4):
        Gamma_b_expected[3 * i:3 * (i + 1), 3 * i:3 * (i + 1)] = R_b_expected
    assert np.allclose(Gamma_b, Gamma_b_expected, atol=1e-12)
    assert np.allclose(Gamma_b @ Gamma_b.T, I12, atol=1e-12)
    assert np.isclose(np.linalg.det(R_b), 1.0, atol=1e-12)
    assert np.allclose(np.cross(R_b[0], R_b[1]), R_b[2], atol=1e-12)

def test_beam_transformation_matrix_error_messages(fcn):
    """
    Test that the function raises ValueError for invalid inputs:
    1. Non-unit reference vector.
    2. Reference vector parallel to the beam axis.
    3. Zero-length beam (identical start and end coordinates).
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, reference_vector=np.array([0.0, 0.0, 2.0]))
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 5.0, 0.0, 0.0, reference_vector=np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        fcn(1.0, 2.0, 3.0, 1.0, 2.0, 3.0, reference_vector=None)
def test_cardinal_axis_alignment(fcn):
    """
    Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """

    def normalize(v):
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        assert n > 0
        return v / n

    def expected_R_from_beam(p1, p2, reference_vector):
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        ex = normalize(p2 - p1)
        if reference_vector is None:
            ez = np.array([0.0, 0.0, 1.0])
            ey = np.array([0.0, 1.0, 0.0])
            if np.isclose(abs(np.dot(ex, ez)), 1.0, atol=1e-12):
                ref = ey
            else:
                ref = ez
        else:
            ref = normalize(reference_vector)
        ey_local = normalize(np.cross(ref, ex))
        ez_local = normalize(np.cross(ex, ey_local))
        return np.column_stack((ex, ey_local, ez_local))

    def R_matches_expected(R, expected):
        return np.allclose(R, expected, atol=1e-12, rtol=0) or np.allclose(R, expected.T, atol=1e-12, rtol=0)
    p1 = [0.0, 0.0, 0.0]
    p2 = [1.0, 0.0, 0.0]
    Gamma = fcn(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], None)
    assert Gamma.shape == (12, 12)
    R = Gamma[:3, :3]
    expected_R = expected_R_from_beam(p1, p2, None)
    assert R_matches_expected(R, expected_R)
    p2 = [0.0, 1.0, 0.0]
    Gamma = fcn(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], None)
    assert Gamma.shape == (12, 12)
    R = Gamma[:3, :3]
    expected_R = expected_R_from_beam(p1, p2, None)
    assert R_matches_expected(R, expected_R)
    p2 = [0.0, 0.0, 1.0]
    Gamma = fcn(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], None)
    assert Gamma.shape == (12, 12)
    R = Gamma[:3, :3]
    expected_R = expected_R_from_beam(p1, p2, None)
    assert R_matches_expected(R, expected_R)

def test_transformation_matrix_properties(fcn):
    """
    Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct.
    """

    def normalize(v):
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        assert n > 0
        return v / n

    def get_R(Gamma):
        return Gamma[:3, :3]

    def check_block_structure(Gamma, R):
        for i in (0, 3, 6, 9):
            assert np.allclose(Gamma[i:i + 3, i:i + 3], R, atol=1e-12, rtol=0)
        for i in (0, 3, 6, 9):
            for j in (0, 3, 6, 9):
                if i != j:
                    assert np.allclose(Gamma[i:i + 3, j:j + 3], 0.0, atol=1e-12, rtol=0)

    def choose_orientation(R, ex_expected):
        d_cols = np.linalg.norm(R[:, 0] - ex_expected)
        d_rows = np.linalg.norm(R[0, :] - ex_expected)
        if d_cols <= d_rows:
            return 'cols'
        return 'rows'
    p1 = np.array([1.0, 2.0, 3.0])
    p2 = np.array([4.0, 6.0, 9.0])
    Gamma = fcn(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], None)
    assert Gamma.shape == (12, 12)
    R = get_R(Gamma)
    I3 = np.eye(3)
    assert np.allclose(R.T @ R, I3, atol=1e-12, rtol=0)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)
    check_block_structure(Gamma, R)
    I12 = np.eye(12)
    assert np.allclose(Gamma.T @ Gamma, I12, atol=1e-12, rtol=0)
    ex_expected = normalize(p2 - p1)
    mode = choose_orientation(R, ex_expected)
    if mode == 'cols':
        (lx, ly, lz) = (R[:, 0], R[:, 1], R[:, 2])
    else:
        (lx, ly, lz) = (R[0, :], R[1, :], R[2, :])
    assert np.allclose(lx, ex_expected, atol=1e-12, rtol=0)
    ref_z = np.array([0.0, 0.0, 1.0])
    ly_expected = normalize(np.cross(ref_z, ex_expected))
    lz_expected = normalize(np.cross(ex_expected, ly_expected))
    assert np.allclose(ly, ly_expected, atol=1e-12, rtol=0)
    assert np.allclose(lz, lz_expected, atol=1e-12, rtol=0)
    assert np.allclose(np.cross(lx, ly), lz, atol=1e-12, rtol=0)
    p1_v = np.array([0.0, 1.0, 2.0])
    p2_v = np.array([0.0, 1.0, 5.0])
    Gamma_v = fcn(p1_v[0], p1_v[1], p1_v[2], p2_v[0], p2_v[1], p2_v[2], None)
    Rv = get_R(Gamma_v)
    assert np.allclose(Rv.T @ Rv, I3, atol=1e-12, rtol=0)
    assert np.isclose(np.linalg.det(Rv), 1.0, atol=1e-12)
    ex_v = normalize(p2_v - p1_v)
    mode_v = choose_orientation(Rv, ex_v)
    if mode_v == 'cols':
        (lx_v, ly_v, lz_v) = (Rv[:, 0], Rv[:, 1], Rv[:, 2])
    else:
        (lx_v, ly_v, lz_v) = (Rv[0, :], Rv[1, :], Rv[2, :])
    assert np.allclose(lx_v, ex_v, atol=1e-12, rtol=0)
    ref_y = np.array([0.0, 1.0, 0.0])
    ly_v_expected = normalize(np.cross(ref_y, ex_v))
    lz_v_expected = normalize(np.cross(ex_v, ly_v_expected))
    assert np.allclose(ly_v, ly_v_expected, atol=1e-12, rtol=0)
    assert np.allclose(lz_v, lz_v_expected, atol=1e-12, rtol=0)
    assert np.allclose(np.cross(lx_v, ly_v), lz_v, atol=1e-12, rtol=0)
    check_block_structure(Gamma_v, Rv)
    assert np.allclose(Gamma_v.T @ Gamma_v, I12, atol=1e-12, rtol=0)
    p1_c = np.array([0.0, 0.0, 0.0])
    p2_c = np.array([2.0, 0.0, 0.0])
    ex_c = normalize(p2_c - p1_c)
    ref_custom = normalize(np.array([0.0, 1.0, 1.0]))
    ey_c = normalize(np.cross(ref_custom, ex_c))
    ez_c = normalize(np.cross(ex_c, ey_c))
    expected_c = np.column_stack((ex_c, ey_c, ez_c))
    Gamma_c = fcn(p1_c[0], p1_c[1], p1_c[2], p2_c[0], p2_c[1], p2_c[2], ref_custom)
    Rc = get_R(Gamma_c)
    assert np.allclose(Rc.T @ Rc, I3, atol=1e-12, rtol=0)
    assert np.isclose(np.linalg.det(Rc), 1.0, atol=1e-12)
    assert np.allclose(Gamma_c.T @ Gamma_c, I12, atol=1e-12, rtol=0)
    assert np.allclose(Rc, expected_c, atol=1e-12, rtol=0) or np.allclose(Rc, expected_c.T, atol=1e-12, rtol=0)
    check_block_structure(Gamma_c, Rc)

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
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.array([0.0, 0.0, 2.0]))
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None)
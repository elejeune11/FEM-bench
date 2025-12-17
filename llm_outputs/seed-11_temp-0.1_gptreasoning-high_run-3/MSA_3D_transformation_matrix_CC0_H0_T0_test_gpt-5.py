def test_cardinal_axis_alignment(fcn):
    """
    Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    i = np.array([1.0, 0.0, 0.0])
    j = np.array([0.0, 1.0, 0.0])
    k = np.array([0.0, 0.0, 1.0])

    def expected_R(p1, p2, ref):
        ex = p2 - p1
        ex = ex / np.linalg.norm(ex)
        if ref is None:
            if np.isclose(abs(np.dot(ex, k)), 1.0, atol=1e-12):
                ref_vec = j
            else:
                ref_vec = k
        else:
            ref_vec = ref
        ey = np.cross(ref_vec, ex)
        ey = ey / np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        ez = ez / np.linalg.norm(ez)
        return np.vstack((ex, ey, ez))
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    Gamma_x = fcn(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], k)
    R_x = Gamma_x[:3, :3]
    R_x_exp = expected_R(p1, p2, k)
    assert R_x.shape == (3, 3)
    assert np.allclose(R_x, R_x_exp, atol=1e-12, rtol=1e-09)
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])
    Gamma_y = fcn(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], k)
    R_y = Gamma_y[:3, :3]
    R_y_exp = expected_R(p1, p2, k)
    assert R_y.shape == (3, 3)
    assert np.allclose(R_y, R_y_exp, atol=1e-12, rtol=1e-09)
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([0.0, 0.0, 1.0])
    Gamma_z = fcn(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], None)
    R_z = Gamma_z[:3, :3]
    R_z_exp = expected_R(p1, p2, None)
    assert R_z.shape == (3, 3)
    assert np.allclose(R_z, R_z_exp, atol=1e-12, rtol=1e-09)

def test_transformation_matrix_properties(fcn):
    """
    Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Verifies that:
      R @ local_x = e1, R @ local_y = e2, R @ local_z = e3.
    """

    def triad(p1, p2, ref):
        ex = p2 - p1
        ex = ex / np.linalg.norm(ex)
        k = np.array([0.0, 0.0, 1.0])
        j = np.array([0.0, 1.0, 0.0])
        if ref is None:
            if np.isclose(abs(np.dot(ex, k)), 1.0, atol=1e-12):
                ref_vec = j
            else:
                ref_vec = k
        else:
            ref_vec = ref
        ey = np.cross(ref_vec, ex)
        ey = ey / np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        ez = ez / np.linalg.norm(ez)
        return (ex, ey, ez)
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 2.0, 3.0])
    Gamma = fcn(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], None)
    assert Gamma.shape == (12, 12)
    R = Gamma[:3, :3]
    for idx in range(4):
        block = Gamma[3 * idx:3 * (idx + 1), 3 * idx:3 * (idx + 1)]
        assert np.allclose(block, R, atol=1e-12, rtol=1e-09)
    zeros3 = np.zeros((3, 3))
    for i in range(4):
        for j in range(4):
            if i != j:
                sub = Gamma[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)]
                assert np.allclose(sub, zeros3, atol=1e-12, rtol=1e-09)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12, rtol=1e-09)
    detR = np.linalg.det(R)
    assert np.isclose(detR, 1.0, atol=1e-12, rtol=1e-09)
    ex, ey, ez = triad(p1, p2, None)
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    e3 = np.array([0.0, 0.0, 1.0])
    assert np.allclose(R @ ex, e1, atol=1e-12, rtol=1e-09)
    assert np.allclose(R @ ey, e2, atol=1e-12, rtol=1e-09)
    assert np.allclose(R @ ez, e3, atol=1e-12, rtol=1e-09)
    assert np.allclose(Gamma.T @ Gamma, np.eye(12), atol=1e-12, rtol=1e-09)
    d_global = np.hstack([ex, ex, ex, ex])
    d_local = Gamma @ d_global
    expected_local = np.tile(e1, 4)
    assert np.allclose(d_local, expected_local, atol=1e-12, rtol=1e-09)
    p1v = np.array([2.0, -1.0, 0.0])
    p2v = np.array([2.0, -1.0, 5.0])
    Gv = fcn(p1v[0], p1v[1], p1v[2], p2v[0], p2v[1], p2v[2], None)
    Rv = Gv[:3, :3]
    assert np.allclose(Rv.T @ Rv, np.eye(3), atol=1e-12, rtol=1e-09)
    assert np.isclose(np.linalg.det(Rv), 1.0, atol=1e-12, rtol=1e-09)
    exv, eyv, ezv = triad(p1v, p2v, None)
    assert np.allclose(Rv @ exv, e1, atol=1e-12, rtol=1e-09)
    assert np.allclose(Rv @ eyv, e2, atol=1e-12, rtol=1e-09)
    assert np.allclose(Rv @ ezv, e3, atol=1e-12, rtol=1e-09)
    for idx in range(4):
        block = Gv[3 * idx:3 * (idx + 1), 3 * idx:3 * (idx + 1)]
        assert np.allclose(block, Rv, atol=1e-12, rtol=1e-09)
    assert np.allclose(Gv.T @ Gv, np.eye(12), atol=1e-12, rtol=1e-09)

def test_beam_transformation_matrix_error_messages(fcn):
    """
    Test that the function raises appropriate ValueError exceptions for invalid reference vectors.
    Verifies proper error handling for two invalid reference vector conditions:
    1. Non-unit vector: When the reference vector's magnitude is not equal to 1.0
    2. Parallel reference vector: When the reference vector is parallel to the beam axis
    Also verifies proper error handling for:
    3. Zero-length beam: start and end coordinates are identical
    All conditions should raise ValueError.
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.array([1.0, 1.0, 0.0]))
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 2.0, 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        fcn(1.0, -2.0, 3.0, 1.0, -2.0, 3.0, None)
def test_cardinal_axis_alignment(fcn):
    """
    Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    import numpy as np
    Gx = fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.array([0.0, 0.0, 1.0]))
    assert Gx.shape == (12, 12)
    Rx = Gx[0:3, 0:3]
    assert np.allclose(Rx, np.eye(3), atol=1e-12)
    Gy = fcn(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, np.array([0.0, 0.0, 1.0]))
    assert Gy.shape == (12, 12)
    Ry = Gy[0:3, 0:3]
    expected_Ry = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(Ry, expected_Ry, atol=1e-12)
    Gz = fcn(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, np.array([0.0, 1.0, 0.0]))
    assert Gz.shape == (12, 12)
    Rz = Gz[0:3, 0:3]
    expected_Rz = np.column_stack((np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])))
    assert np.allclose(Rz, expected_Rz, atol=1e-12)
    for i in (0, 3, 6, 9):
        assert np.allclose(Gx[i:i + 3, i:i + 3], Rx, atol=1e-12)

def test_transformation_matrix_properties(fcn):
    """
    Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Checks:
    """
    import numpy as np
    cases = [((1.2, -0.7, 3.4), (-0.6, 2.5, 4.1), None), ((-1.0, 2.0, -3.0), (2.5, -0.5, 1.2), np.array([0.0, 0.0, 1.0])), ((1.0, 1.0, 0.0), (1.0, 1.0, 2.0), None)]
    for ((x1, y1, z1), (x2, y2, z2), ref) in cases:
        Gamma = fcn(x1, y1, z1, x2, y2, z2, ref)
        assert Gamma.shape == (12, 12)
        R = Gamma[0:3, 0:3]
        p1 = np.array([x1, y1, z1], dtype=float)
        p2 = np.array([x2, y2, z2], dtype=float)
        local_x = p2 - p1
        L = np.linalg.norm(local_x)
        assert L > 0.0
        local_x = local_x / L
        if ref is None:
            if abs(abs(local_x[2]) - 1.0) < 1e-12:
                ref_vec = np.array([0.0, 1.0, 0.0])
            else:
                ref_vec = np.array([0.0, 0.0, 1.0])
        else:
            ref_vec = np.asarray(ref, dtype=float)
        local_y = np.cross(ref_vec, local_x)
        ny = np.linalg.norm(local_y)
        assert ny > 0.0
        local_y = local_y / ny
        local_z = np.cross(local_x, local_y)
        nz = np.linalg.norm(local_z)
        assert nz > 0.0
        local_z = local_z / nz
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
        e3 = np.array([0.0, 0.0, 1.0])
        assert np.allclose(R @ e1, local_x, atol=1e-12)
        assert np.allclose(R @ e2, local_y, atol=1e-12)
        assert np.allclose(R @ e3, local_z, atol=1e-12)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)
        for i in (0, 3, 6, 9):
            assert np.allclose(Gamma[i:i + 3, i:i + 3], R, atol=1e-12)
        I12 = np.eye(12)
        assert np.allclose(Gamma.T @ Gamma, I12, atol=1e-12)
        assert np.allclose(Gamma @ Gamma.T, I12, atol=1e-12)
        rng = np.random.RandomState(42)
        v_local = rng.randn(12)
        v_global = Gamma @ v_local
        v_back = Gamma.T @ v_global
        assert np.allclose(v_back, v_local, atol=1e-12)

def test_beam_transformation_matrix_error_messages(fcn):
    """
    Test that the function raises appropriate ValueError exceptions for invalid reference vectors.
    Verifies proper error handling for:
    1. Non-unit vector: When the reference vector's magnitude is not equal to 1.0
    2. Parallel reference vector: When the reference vector is parallel to the beam axis
       (resulting in a zero cross product and undefined local axes)
    3. Zero-length beam: start and end coordinates are identical
    All conditions should raise ValueError.
    """
    import numpy as np
    import pytest
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.array([0.0, 0.0, 2.0]))
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.array([0.0, 0.0, 1.0]))
def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    import numpy as np
    gamma = fcn(0, 0, 0, 1, 0, 0, None)
    assert np.allclose(gamma[0:3, 0:3], np.eye(3))
    assert np.allclose(gamma[3:6, 3:6], np.eye(3))
    assert np.allclose(gamma[6:9, 6:9], np.eye(3))
    assert np.allclose(gamma[9:12, 9:12], np.eye(3))
    gamma = fcn(0, 0, 0, 0, 1, 0, None)
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert np.allclose(gamma[0:3, 0:3], R)
    assert np.allclose(gamma[3:6, 3:6], R)
    assert np.allclose(gamma[6:9, 6:9], R)
    assert np.allclose(gamma[9:12, 9:12], R)
    gamma = fcn(0, 0, 0, 0, 0, 1, None)
    R = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    assert np.allclose(gamma[0:3, 0:3], R)
    assert np.allclose(gamma[3:6, 3:6], R)
    assert np.allclose(gamma[6:9, 6:9], R)
    assert np.allclose(gamma[9:12, 9:12], R)

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct."""
    import numpy as np
    gamma = fcn(0, 0, 0, 1, 1, 1, None)
    assert np.allclose(gamma @ gamma.T, np.eye(12))
    assert np.allclose(gamma.T @ gamma, np.eye(12))
    assert np.allclose(np.linalg.det(gamma[0:3, 0:3]), 1.0)
    assert np.allclose(np.linalg.det(gamma[3:6, 3:6]), 1.0)
    assert np.allclose(np.linalg.det(gamma[6:9, 6:9]), 1.0)
    assert np.allclose(np.linalg.det(gamma[9:12, 9:12]), 1.0)
    for i in range(4):
        for j in range(4):
            if i != j:
                assert np.allclose(gamma[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)], np.zeros((3, 3)))

def test_beam_transformation_matrix_error_messages(fcn):
    """Test that the function raises appropriate ValueError exceptions for invalid reference vectors.
    Verifies proper error handling for two invalid reference vector conditions:
    1. Non-unit vector: When the reference vector's magnitude is not equal to 1.0
    2. Parallel reference vector: When the reference vector is parallel to the beam axis
    Also veriefies proper error handling for:
    3. Zero-length beam: start and end coordinates are identical
    All conditions should raise ValueError"""
    import numpy as np
    import pytest
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, np.array([2, 0, 0]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0, 0]))
    with pytest.raises(ValueError):
        fcn(1, 1, 1, 1, 1, 1, None)
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0, 0, 0]))
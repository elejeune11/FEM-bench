def test_element_stiffness_comprehensive(fcn):
    """Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.
    This test checks the following properties of the stiffness matrix computed by
    `element_stiffness_linear_elastic_1D` for a two-node linear element:
    1. Analytical correctness:
          (EA/L) * [[1, -1], [-1, 1]]
    2. Shape and symmetry:
    3. Singularity:
          reflecting rigid body motion.
    4. Integration consistency:
          Gauss quadrature rules when applied to linear elements, since exact integration is achieved.
    Note: Minor floating-point differences may arise due to roundoff when summing weighted values.
    This test uses a strict but reasonable tolerance to allow for numerical consistency considering the limitations of floating point arithmetic.
    """
    E = 200000000000.0
    A = 0.01
    L = 2.0
    x_elem = np.array([0.0, L])
    K_2pt = fcn(x_elem, E, A, n_gauss=2)
    K_expected = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(K_2pt, K_expected, rtol=1e-10, atol=1e-12), 'Stiffness matrix does not match analytical solution'
    assert K_2pt.shape == (2, 2), 'Stiffness matrix must be 2x2'
    assert np.allclose(K_2pt, K_2pt.T, rtol=1e-10, atol=1e-12), 'Stiffness matrix must be symmetric'
    det_K = np.linalg.det(K_2pt)
    assert abs(det_K) < 1e-10, 'Stiffness matrix should be singular (zero determinant)'
    K_1pt = fcn(x_elem, E, A, n_gauss=1)
    K_3pt = fcn(x_elem, E, A, n_gauss=3)
    assert np.allclose(K_1pt, K_2pt, rtol=1e-10, atol=1e-12), '1-point Gauss quadrature inconsistent'
    assert np.allclose(K_2pt, K_3pt, rtol=1e-10, atol=1e-12), '3-point Gauss quadrature inconsistent'
    assert np.allclose(K_1pt, K_3pt, rtol=1e-10, atol=1e-12), '1-point and 3-point quadrature inconsistent'

def test_element_stiffness_zero_length(fcn):
    """Test behavior for zero-length element (edge case)."""
    E = 200000000000.0
    A = 0.01
    x_elem = np.array([1.0, 1.0])
    with pytest.raises((ZeroDivisionError, ValueError, RuntimeError)):
        fcn(x_elem, E, A, n_gauss=2)

def test_element_stiffness_negative_length(fcn):
    """Test behavior for negative length element (invalid input)."""
    E = 200000000000.0
    A = 0.01
    x_elem = np.array([2.0, 1.0])
    try:
        K = fcn(x_elem, E, A, n_gauss=2)
        assert K[0, 0] > 0 and K[1, 1] > 0
    except (ValueError, RuntimeError):
        pass

def test_element_stiffness_material_properties(fcn):
    """Test sensitivity to material properties."""
    L = 2.0
    x_elem = np.array([0.0, L])
    test_cases = [(100000000000.0, 0.005), (500000000000.0, 0.02), (1.0, 1.0)]
    for (E, A) in test_cases:
        K = fcn(x_elem, E, A, n_gauss=2)
        K_expected = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
        assert np.allclose(K, K_expected, rtol=1e-10, atol=1e-12), f'Failed for E={E}, A={A}'

def test_element_stiffness_different_positions(fcn):
    """Test that position offset doesn't affect result (translation invariance)."""
    E = 200000000000.0
    A = 0.01
    L = 2.0
    test_positions = [np.array([0.0, L]), np.array([5.0, 5.0 + L]), np.array([-3.0, -3.0 + L])]
    reference_K = fcn(test_positions[0], E, A, n_gauss=2)
    for x_elem in test_positions[1:]:
        K = fcn(x_elem, E, A, n_gauss=2)
        assert np.allclose(K, reference_K, rtol=1e-10, atol=1e-12), 'Stiffness matrix should be translation invariant'
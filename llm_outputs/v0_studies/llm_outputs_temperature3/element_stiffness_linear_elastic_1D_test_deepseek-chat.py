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
    x_elem = np.array([0.0, 2.0])
    E = 100.0
    A = 0.5
    L = x_elem[1] - x_elem[0]
    expected_k = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    k_2 = fcn(x_elem, E, A, 2)
    assert np.allclose(k_2, expected_k, rtol=1e-10, atol=1e-12)
    assert k_2.shape == (2, 2)
    assert np.allclose(k_2, k_2.T, rtol=1e-10, atol=1e-12)
    det_k = np.linalg.det(k_2)
    assert abs(det_k) < 1e-10
    k_1 = fcn(x_elem, E, A, 1)
    k_3 = fcn(x_elem, E, A, 3)
    assert np.allclose(k_1, k_2, rtol=1e-10, atol=1e-12)
    assert np.allclose(k_3, k_2, rtol=1e-10, atol=1e-12)

def test_element_stiffness_different_lengths(fcn):
    """Test the element stiffness matrix for elements of different lengths."""
    E = 200.0
    A = 1.0
    x_elem1 = np.array([0.0, 1.0])
    k1 = fcn(x_elem1, E, A, 2)
    expected_k1 = E * A / 1.0 * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(k1, expected_k1, rtol=1e-10, atol=1e-12)
    x_elem2 = np.array([0.0, 5.0])
    k2 = fcn(x_elem2, E, A, 2)
    expected_k2 = E * A / 5.0 * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(k2, expected_k2, rtol=1e-10, atol=1e-12)
    x_elem3 = np.array([3.0, 8.0])
    k3 = fcn(x_elem3, E, A, 2)
    expected_k3 = E * A / 5.0 * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(k3, expected_k3, rtol=1e-10, atol=1e-12)

def test_element_stiffness_material_properties(fcn):
    """Test the element stiffness matrix for different material properties."""
    x_elem = np.array([0.0, 2.0])
    k_high = fcn(x_elem, E=1000.0, A=1.0, n_gauss=2)
    expected_high = 1000.0 * 1.0 / 2.0 * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(k_high, expected_high, rtol=1e-10, atol=1e-12)
    k_large_area = fcn(x_elem, E=100.0, A=10.0, n_gauss=2)
    expected_large_area = 100.0 * 10.0 / 2.0 * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(k_large_area, expected_large_area, rtol=1e-10, atol=1e-12)
    k_small = fcn(x_elem, E=0.1, A=0.01, n_gauss=2)
    expected_small = 0.1 * 0.01 / 2.0 * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(k_small, expected_small, rtol=1e-10, atol=1e-12)

def test_element_stiffness_symmetry_positive_definite(fcn):
    """Test that the stiffness matrix is symmetric and positive semi-definite."""
    x_elem = np.array([0.0, 3.0])
    E = 150.0
    A = 2.0
    k = fcn(x_elem, E, A, 2)
    assert np.allclose(k, k.T, rtol=1e-10, atol=1e-12)
    eigenvalues = np.linalg.eigvals(k)
    assert np.all(eigenvalues >= -1e-12)
    assert abs(eigenvalues[0]) < 1e-10 or abs(eigenvalues[1]) < 1e-10
    assert eigenvalues[0] > 1e-10 or eigenvalues[1] > 1e-10

def test_element_stiffness_gauss_points(fcn):
    """Test that different numbers of Gauss points produce consistent results for linear elements."""
    x_elem = np.array([1.0, 4.0])
    E = 75.0
    A = 1.5
    k_1 = fcn(x_elem, E, A, 1)
    k_2 = fcn(x_elem, E, A, 2)
    k_3 = fcn(x_elem, E, A, 3)
    k_4 = fcn(x_elem, E, A, 4)
    expected_k = E * A / 3.0 * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(k_1, expected_k, rtol=1e-10, atol=1e-12)
    assert np.allclose(k_2, expected_k, rtol=1e-10, atol=1e-12)
    assert np.allclose(k_3, expected_k, rtol=1e-10, atol=1e-12)
    assert np.allclose(k_4, expected_k, rtol=1e-10, atol=1e-12)
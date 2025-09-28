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
    x_elem = np.array([0.0, 2.0])
    L = x_elem[1] - x_elem[0]
    expected_k = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    k_2pt = fcn(x_elem, E, A, 2)
    assert k_2pt.shape == (2, 2), 'Stiffness matrix must be 2x2'
    assert np.allclose(k_2pt, k_2pt.T), 'Stiffness matrix must be symmetric'
    assert np.allclose(k_2pt, expected_k, rtol=1e-10), 'Stiffness matrix does not match analytical solution'
    det_k = np.linalg.det(k_2pt)
    assert abs(det_k) < 1e-10, 'Stiffness matrix should be singular (zero determinant)'
    k_1pt = fcn(x_elem, E, A, 1)
    k_3pt = fcn(x_elem, E, A, 3)
    assert np.allclose(k_1pt, k_2pt, rtol=1e-14), '1-point and 2-point Gauss quadrature should yield identical results'
    assert np.allclose(k_2pt, k_3pt, rtol=1e-14), '2-point and 3-point Gauss quadrature should yield identical results'
    assert np.allclose(k_1pt, k_3pt, rtol=1e-14), '1-point and 3-point Gauss quadrature should yield identical results'

def test_element_stiffness_different_lengths(fcn):
    """Test the stiffness matrix for elements with different lengths."""
    E = 100.0
    A = 1.0
    test_lengths = [0.5, 1.0, 2.0, 5.0, 10.0]
    for L in test_lengths:
        x_elem = np.array([0.0, L])
        k = fcn(x_elem, E, A, 2)
        expected_k = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
        assert np.allclose(k, expected_k, rtol=1e-10), f'Stiffness incorrect for length L={L}'
        assert abs(k[0, 0] - E * A / L) < 1e-10, f'Diagonal term should be EA/L for L={L}'

def test_element_stiffness_material_properties(fcn):
    """Test the stiffness matrix for different material properties."""
    x_elem = np.array([0.0, 1.0])
    test_cases = [(100.0, 1.0), (200.0, 1.0), (100.0, 2.0), (50.0, 0.5), (1000.0, 0.1)]
    for (E, A) in test_cases:
        k = fcn(x_elem, E, A, 2)
        expected_k = E * A * np.array([[1.0, -1.0], [-1.0, 1.0]])
        assert np.allclose(k, expected_k, rtol=1e-10), f'Stiffness incorrect for E={E}, A={A}'
        assert abs(k[0, 0] - E * A) < 1e-10, f'Diagonal term should be E*A for unit length'

def test_element_stiffness_negative_coordinates(fcn):
    """Test the stiffness matrix with negative nodal coordinates."""
    E = 100.0
    A = 1.0
    test_cases = [np.array([-1.0, 1.0]), np.array([-2.0, -1.0]), np.array([5.0, 10.0]), np.array([-3.0, 2.0])]
    for x_elem in test_cases:
        L = x_elem[1] - x_elem[0]
        k = fcn(x_elem, E, A, 2)
        expected_k = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
        assert np.allclose(k, expected_k, rtol=1e-10), f'Stiffness incorrect for coordinates {x_elem}'
        assert L > 0, 'Element length must be positive'
        assert k[0, 0] > 0, 'Diagonal stiffness term must be positive'

def test_element_stiffness_edge_cases(fcn):
    """Test edge cases and potential error conditions."""
    E = 100.0
    A = 1.0
    x_small = np.array([0.0, 1e-10])
    k_small = fcn(x_small, E, A, 2)
    x_large = np.array([0.0, 10000000000.0])
    k_large = fcn(x_large, E, A, 2)
    assert k_small[0, 0] > k_large[0, 0], 'Smaller elements should have higher stiffness'
    x_unit = np.array([0.0, 1.0])
    k_unit = fcn(x_unit, E, A, 2)
    assert abs(k_small[0, 0] * 1e-10 - E * A) < 1e-05, 'Stiffness should scale as 1/L'
    assert abs(k_large[0, 0] * 10000000000.0 - E * A) < 1e-05, 'Stiffness should scale as 1/L'

def test_element_stiffness_gauss_points(fcn):
    """Test that different numbers of Gauss points produce consistent results."""
    E = 150.0
    A = 0.5
    x_elem = np.array([0.0, 3.0])
    gauss_points = [1, 2, 3, 4, 5]
    results = []
    for n_gauss in gauss_points:
        k = fcn(x_elem, E, A, n_gauss)
        results.append(k)
        assert k.shape == (2, 2), f'Result for {n_gauss} Gauss points must be 2x2'
        assert np.allclose(k, k.T), f'Result for {n_gauss} Gauss points must be symmetric'
    for i in range(1, len(results)):
        assert np.allclose(results[0], results[i], rtol=1e-14), f'Results for {gauss_points[0]} and {gauss_points[i]} Gauss points should be identical'
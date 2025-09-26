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
    expected_factor = E * A / L
    expected_matrix = expected_factor * np.array([[1.0, -1.0], [-1.0, 1.0]])
    results = {}
    for n_gauss in [1, 2, 3]:
        K = fcn(x_elem, E, A, n_gauss)
        results[n_gauss] = K
        assert K.shape == (2, 2), f'Stiffness matrix for n_gauss={n_gauss} must be 2x2'
        assert np.allclose(K, K.T, atol=1e-12), f'Stiffness matrix for n_gauss={n_gauss} must be symmetric'
        det = np.linalg.det(K)
        assert abs(det) < 1e-12, f'Stiffness matrix for n_gauss={n_gauss} must be singular (det={det})'
        assert np.allclose(K, expected_matrix, atol=1e-12), f"Stiffness matrix for n_gauss={n_gauss} doesn't match analytical solution"
    assert np.allclose(results[1], results[2], atol=1e-12), 'Results for 1-point and 2-point Gauss quadrature must match'
    assert np.allclose(results[1], results[3], atol=1e-12), 'Results for 1-point and 3-point Gauss quadrature must match'
    assert np.allclose(results[2], results[3], atol=1e-12), 'Results for 2-point and 3-point Gauss quadrature must match'
def test_element_stiffness_comprehensive(fcn):
    """
    Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.
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
    E = 210000000000.0
    A = 0.005
    x_elem = np.array([0.2, 1.0])
    L = x_elem[1] - x_elem[0]
    k_analytical = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    k_results = {}
    for n_gauss in [1, 2, 3]:
        k_computed = fcn(x_elem, E, A, n_gauss)
        k_results[n_gauss] = k_computed
        np.testing.assert_allclose(k_computed, k_analytical, rtol=1e-14, atol=1e-14)
        assert k_computed.shape == (2, 2)
        np.testing.assert_allclose(k_computed, k_computed.T, rtol=1e-14, atol=1e-14)
        assert np.isclose(np.linalg.det(k_computed), 0.0, atol=1e-09)
    np.testing.assert_allclose(k_results[1], k_results[2], rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(k_results[2], k_results[3], rtol=1e-15, atol=1e-15)
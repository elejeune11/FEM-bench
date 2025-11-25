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
    E = 210000000000.0
    A = 0.00015
    x_elem = np.array([0.5, 2.5])
    L = x_elem[1] - x_elem[0]
    k_analytical = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    gauss_points_to_test = [1, 2, 3]
    results = []
    for n_gauss in gauss_points_to_test:
        k_computed = fcn(x_elem=x_elem, E=E, A=A, n_gauss=n_gauss)
        assert k_computed.shape == (2, 2), f'Shape check failed for n_gauss={n_gauss}'
        assert np.allclose(k_computed, k_computed.T), f'Symmetry check failed for n_gauss={n_gauss}'
        assert np.isclose(np.linalg.det(k_computed), 0.0), f'Singularity check failed for n_gauss={n_gauss}'
        assert np.allclose(k_computed, k_analytical), f'Analytical correctness check failed for n_gauss={n_gauss}'
        results.append(k_computed)
    for i in range(len(results) - 1):
        assert np.allclose(results[i], results[i + 1]), f'Integration consistency failed between n_gauss={gauss_points_to_test[i]} and n_gauss={gauss_points_to_test[i + 1]}'
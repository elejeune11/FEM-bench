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
    k_1 = fcn(x_elem, E, A, 1)
    k_2 = fcn(x_elem, E, A, 2)
    k_3 = fcn(x_elem, E, A, 3)
    assert k_1.shape == (2, 2)
    assert k_2.shape == (2, 2)
    assert k_3.shape == (2, 2)
    assert np.allclose(k_1, k_2, rtol=1e-14, atol=1e-14)
    assert np.allclose(k_2, k_3, rtol=1e-14, atol=1e-14)
    assert np.allclose(k_1, expected_k, rtol=1e-14, atol=1e-14)
    assert np.allclose(k_1, k_1.T, rtol=1e-14, atol=1e-14)
    assert np.allclose(k_2, k_2.T, rtol=1e-14, atol=1e-14)
    assert np.allclose(k_3, k_3.T, rtol=1e-14, atol=1e-14)
    assert abs(np.linalg.det(k_1)) < 1e-14
    assert abs(np.linalg.det(k_2)) < 1e-14
    assert abs(np.linalg.det(k_3)) < 1e-14
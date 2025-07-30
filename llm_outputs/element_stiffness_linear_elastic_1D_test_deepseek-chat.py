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
    """
    E = 100.0
    A = 2.0
    L = 5.0
    x_elem = np.array([0.0, L])
    tol = 1e-10
    expected = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    K = fcn(x_elem, E, A, 2)
    assert K.shape == (2, 2)
    assert np.allclose(K, K.T, atol=tol)
    assert np.allclose(K, expected, atol=tol)
    assert np.abs(np.linalg.det(K)) < tol
    K1 = fcn(x_elem, E, A, 1)
    K2 = fcn(x_elem, E, A, 2)
    K3 = fcn(x_elem, E, A, 3)
    assert np.allclose(K1, K2, atol=tol)
    assert np.allclose(K1, K3, atol=tol)
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
    """
    E = 1.0
    A = 1.0
    x_elem = np.array([0.0, 1.0])
    L = x_elem[1] - x_elem[0]
    K_analytical = E * A / L * np.array([[1, -1], [-1, 1]])
    K_computed = fcn(x_elem, E, A, 2)
    assert np.allclose(K_computed, K_analytical)
    assert K_computed.shape == (2, 2)
    assert np.allclose(K_computed, K_computed.T)
    assert np.isclose(np.linalg.det(K_computed), 0)
    for n_gauss in [1, 2, 3]:
        K = fcn(x_elem, E, A, n_gauss)
        assert np.allclose(K, K_computed)
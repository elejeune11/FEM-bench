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
    import numpy as np
    x_elem = np.array([1.5, 3.5], dtype=float)
    E = 16.0
    A = 0.5
    L = x_elem[1] - x_elem[0]
    assert L > 0.0
    k = E * A / L
    expected = k * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
    K1 = fcn(x_elem, E, A, 1)
    K2 = fcn(x_elem, E, A, 2)
    K3 = fcn(x_elem, E, A, 3)
    for K in (K1, K2, K3):
        assert isinstance(K, np.ndarray)
        assert K.shape == (2, 2)
        assert np.allclose(K, K.T, rtol=1e-12, atol=1e-12)
    assert np.allclose(K2, expected, rtol=1e-12, atol=1e-12)
    detK2 = float(np.linalg.det(K2))
    det_tol = 1e-12 * max(1.0, np.linalg.norm(expected, ord='fro') ** 2)
    assert abs(detK2) <= det_tol
    assert np.allclose(K1, K2, rtol=1e-12, atol=1e-12)
    assert np.allclose(K3, K2, rtol=1e-12, atol=1e-12)
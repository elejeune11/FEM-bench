def test_element_stiffness_comprehensive(fcn):
    """
    Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.
    This test checks the following properties of the stiffness matrix computed by
    element_stiffness_linear_elastic_1D for a two-node linear element:
    1. Analytical correctness:
          (EA/L) * [[1, -1], [-1, 1]]
    2. Shape and symmetry:
    3. Singularity:
          reflecting rigid body motion.
    4. Integration consistency:
          Gauss quadrature rules when applied to linear elements, since exact integration is achieved.
    """
    import numpy as np
    E = 200000000000.0
    A = 0.01
    x = np.array([0.0, 2.0])
    L = x[1] - x[0]
    k_exact = E * A / L * np.array([[1, -1], [-1, 1]])
    k = fcn(x, E, A, n_gauss=2)
    assert np.allclose(k, k_exact, rtol=1e-13, atol=1e-13)
    assert k.shape == (2, 2)
    assert np.allclose(k, k.T, rtol=1e-13, atol=1e-13)
    assert abs(np.linalg.det(k)) < 1e-10
    k1 = fcn(x, E, A, n_gauss=1)
    k2 = fcn(x, E, A, n_gauss=2)
    k3 = fcn(x, E, A, n_gauss=3)
    assert np.allclose(k1, k2, rtol=1e-13, atol=1e-13)
    assert np.allclose(k2, k3, rtol=1e-13, atol=1e-13)
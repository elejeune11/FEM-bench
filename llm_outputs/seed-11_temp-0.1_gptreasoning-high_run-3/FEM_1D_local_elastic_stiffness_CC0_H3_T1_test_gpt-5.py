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
    import numpy as np
    L = 1.7
    E = 70000000000.0
    A = 0.0034
    x_elem = np.array([2.0, 2.0 + L], dtype=float)
    scale = abs(E * A / L)
    expected = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
    rtol = 1e-12
    atol = rtol * scale
    ks = []
    for n_gauss in (1, 2, 3):
        k = fcn(x_elem, E, A, n_gauss)
        ks.append(k)
        assert isinstance(k, np.ndarray)
        assert k.shape == (2, 2)
        assert np.allclose(k, k.T, rtol=rtol, atol=atol)
        assert np.allclose(k, expected, rtol=rtol, atol=atol)
    k1, k2, k3 = ks
    assert np.allclose(k1, k2, rtol=rtol, atol=atol)
    assert np.allclose(k1, k3, rtol=rtol, atol=atol)
    assert np.allclose(k2, k3, rtol=rtol, atol=atol)
    det_tol = 1e-08 * (E * A / L) ** 2
    det_k2 = float(np.linalg.det(k2))
    assert abs(det_k2) <= det_tol
    rb_mode = np.array([1.0, 1.0])
    rb_residual = k2 @ rb_mode
    rb_tol = 1e-12 * scale
    assert np.linalg.norm(rb_residual, ord=np.inf) <= rb_tol
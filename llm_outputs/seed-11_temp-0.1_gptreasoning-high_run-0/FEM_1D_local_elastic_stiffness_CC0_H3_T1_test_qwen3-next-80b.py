def test_element_stiffness_comprehensive(fcn):
    import numpy as np
    x_elem = np.array([0.0, 1.0])
    E = 210000000000.0
    A = 0.01
    n_gauss_list = [1, 2, 3]
    L = x_elem[1] - x_elem[0]
    K_analytical = E * A / L * np.array([[1, -1], [-1, 1]])
    K_list = [fcn(x_elem, E, A, n_gauss) for n_gauss in n_gauss_list]
    for K in K_list:
        assert K.shape == (2, 2), 'Stiffness matrix must be 2x2'
        assert np.allclose(K, K.T), 'Stiffness matrix must be symmetric'
        assert abs(np.linalg.det(K)) < 1e-12, 'Stiffness matrix must be singular'
    for K in K_list:
        assert np.allclose(K, K_analytical, atol=1e-10, rtol=1e-10), 'Stiffness matrix does not match analytical solution'
    for i in range(len(K_list)):
        for j in range(i + 1, len(K_list)):
            assert np.allclose(K_list[i], K_list[j], atol=1e-12, rtol=1e-12), 'Results must be consistent across Gauss quadrature orders'
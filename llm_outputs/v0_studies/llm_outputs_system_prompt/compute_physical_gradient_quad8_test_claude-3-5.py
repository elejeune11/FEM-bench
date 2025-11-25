def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη², 
    the physical gradient equals the analytic gradient."""
    import numpy as np
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (a, b, c, d, e, f) = (1, 2, 3, 4, 5, 6)
    node_values = np.array([a + b * -1 + c * -1 + d * 1 + e * 1 + f * 1, a + b * 1 + c * -1 + d * 1 - e * 1 + f * 1, a + b * 1 + c * 1 + d * 1 + e * 1 + f * 1, a + b * -1 + c * 1 + d * 1 - e * 1 + f * 1, a + b * 0 + c * -1 + d * 0 + e * 0 + f * 1, a + b * 1 + c * 0 + d * 1 + e * 0 + f * 0, a + b * 0 + c * 1 + d * 0 + e * 0 + f * 1, a + b * -1 + c * 0 + d * 1 + e * 0 + f * 0])
    xi = np.array([-0.5, 0.0, 0.7])
    eta = np.array([0.3, -0.2, 0.4])
    grad_computed = fcn(node_coords, node_values, xi, eta)
    grad_analytic = np.vstack([b + 2 * d * xi + e * eta, c + e * xi + 2 * f * eta])
    assert np.allclose(grad_computed, grad_analytic)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points."""
    import numpy as np
    (R1, R2) = (1.0, 2.0)
    (theta1, theta2) = (0.0, np.pi / 2)

    def polar_to_xy(r, theta):
        return (r * np.cos(theta), r * np.sin(theta))
    node_coords = np.array([polar_to_xy(R1, theta1), polar_to_xy(R2, theta1), polar_to_xy(R2, theta2), polar_to_xy(R1, theta2), polar_to_xy((R1 + R2) / 2, theta1), polar_to_xy(R2, (theta1 + theta2) / 2), polar_to_xy((R1 + R2) / 2, theta2), polar_to_xy(R1, (theta1 + theta2) / 2)])
    (alpha, beta, gamma) = (2, 3, -4)
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    xi = np.array([-0.8, -0.3, 0.0, 0.5, 0.9])
    eta = np.array([0.1, -0.4, 0.2, 0.6, -0.7])
    grad_computed = fcn(node_coords, node_values, xi, eta)
    grad_expected = np.array([[beta], [gamma]]) * np.ones((1, len(xi)))
    assert np.allclose(grad_computed, grad_expected)
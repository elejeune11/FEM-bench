def test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces(fcn):
    """
    Test that the traction integral works on straight edges of a uniformly scaled (2x)
    Q8 element. For each face, apply a constant traction and verify:
    1) The total force equals traction times the physical edge length.
    2) All nodes not on the loaded edge receive zero load.
    """
    node_coords_ref = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    scale = 2.0
    node_coords = scale * node_coords_ref
    traction = np.array([3.0, -5.0], dtype=float)
    edges = {0: (0, 4, 1), 1: (1, 5, 2), 2: (2, 6, 3), 3: (3, 7, 0)}
    for face, edge_nodes in edges.items():
        r_elem = fcn(face, node_coords, traction, num_gauss_pts=2)
        assert isinstance(r_elem, np.ndarray)
        assert r_elem.shape == (16,)
        r_pairs = r_elem.reshape(8, 2)
        total_force = r_pairs[list(edge_nodes), :].sum(axis=0)
        start = node_coords[edge_nodes[0]]
        end = node_coords[edge_nodes[2]]
        L = np.linalg.norm(end - start)
        expected_total = traction * L
        assert np.allclose(total_force, expected_total, rtol=1e-12, atol=1e-12)
        off_edge_nodes = [n for n in range(8) if n not in edge_nodes]
        assert np.allclose(r_pairs[off_edge_nodes, :], 0.0, atol=1e-14)

def test_edl_q8_constant_traction_total_force_on_curved_parabolic_edge(fcn):
    """
    Test curved bottom edge (face=0) with geometry x(s)=s, y(s)=c+k s^2 realized by
    Q8 edge nodes: start=(-1,c+k), mid=(0,c), end=(1,c+k). With constant traction,
    verify the total force equals traction times the exact arc length:
        L_exact = sqrt(1+α) + asinh(sqrt(α)) / sqrt(α),  α = 4 k^2.
    Use 3-point Gauss quadrature on the edge and a reasonable relative tolerance.
    """
    c = 1.5
    k = 0.4
    H = 2.5
    node_coords = np.zeros((8, 2), dtype=float)
    node_coords[0] = [-1.0, c + k]
    node_coords[4] = [0.0, c]
    node_coords[1] = [1.0, c + k]
    node_coords[5] = [1.0, c + k + 0.5 * H]
    node_coords[2] = [1.0, c + k + H]
    node_coords[6] = [0.0, c + k + H]
    node_coords[3] = [-1.0, c + k + H]
    node_coords[7] = [-1.0, c + k + 0.5 * H]
    traction = np.array([1.7, -2.3], dtype=float)
    r_elem = fcn(0, node_coords, traction, num_gauss_pts=3)
    assert isinstance(r_elem, np.ndarray)
    assert r_elem.shape == (16,)
    r_pairs = r_elem.reshape(8, 2)
    edge_nodes = (0, 4, 1)
    total_force = r_pairs[list(edge_nodes), :].sum(axis=0)
    alpha = 4.0 * k * k
    sqrt_alpha = np.sqrt(alpha)
    L_exact = np.sqrt(1.0 + alpha) + np.arcsinh(sqrt_alpha) / sqrt_alpha
    expected_total = traction * L_exact
    assert np.allclose(total_force, expected_total, rtol=0.0005, atol=1e-12)
def test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces(fcn):
    """Test that the traction integral works on straight edge elements scaled by 2x. For each face, the summed nodal forces on the loaded edge equal traction times edge length, and non-edge nodes receive zero load."""
    node_coords = [[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0], [0.0, -2.0], [2.0, 0.0], [0.0, 2.0], [-2.0, 0.0]]
    traction = [2.5, -1.2]
    face_nodes = {0: (0, 4, 1), 1: (1, 5, 2), 2: (2, 6, 3), 3: (3, 7, 0)}
    tol = 1e-12
    zero_tol = 1e-14
    for (face, edge_nodes) in face_nodes.items():
        r_elem = fcn(face, node_coords, traction, 2)
        assert len(r_elem) == 16
        (start, _, end) = edge_nodes
        (x0, y0) = node_coords[start]
        (x1, y1) = node_coords[end]
        L = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        sum_fx = sum((r_elem[2 * i] for i in edge_nodes))
        sum_fy = sum((r_elem[2 * i + 1] for i in edge_nodes))
        exp_fx = traction[0] * L
        exp_fy = traction[1] * L
        denom_x = abs(exp_fx) if abs(exp_fx) > 1.0 else 1.0
        denom_y = abs(exp_fy) if abs(exp_fy) > 1.0 else 1.0
        assert abs(sum_fx - exp_fx) <= tol * denom_x
        assert abs(sum_fy - exp_fy) <= tol * denom_y
        non_edge = [i for i in range(8) if i not in edge_nodes]
        for i in non_edge:
            assert abs(r_elem[2 * i]) <= zero_tol
            assert abs(r_elem[2 * i + 1]) <= zero_tol

def test_edl_q8_constant_traction_total_force_on_curved_parabolic_edge(fcn):
    """Test curved bottom edge with quadratic geometry under constant traction. The total nodal force equals traction times exact arc length within a suitable tolerance; other nodes receive zero load."""
    c = 0.5
    k = 0.3
    top_offset = 2.0
    node_coords = [[-1.0, c + k], [1.0, c + k], [1.0, c + k + top_offset], [-1.0, c + k + top_offset], [0.0, c], [1.0, c + k + top_offset / 2.0], [0.0, c + k + top_offset], [-1.0, c + k + top_offset / 2.0]]
    traction = [1.1, -0.7]
    r_elem = fcn(0, node_coords, traction, 3)
    assert len(r_elem) == 16
    edge_nodes = (0, 4, 1)
    sum_fx = sum((r_elem[2 * i] for i in edge_nodes))
    sum_fy = sum((r_elem[2 * i + 1] for i in edge_nodes))
    alpha = 4.0 * k * k

    def f(s):
        return (1.0 + alpha * s * s) ** 0.5
    N = 20000
    h = 2.0 / N
    s0 = -1.0
    sN = 1.0
    total = f(s0) + f(sN)
    for i in range(1, N):
        s = -1.0 + i * h
        total += (4.0 if i % 2 == 1 else 2.0) * f(s)
    L_exact = h / 3.0 * total
    exp_fx = traction[0] * L_exact
    exp_fy = traction[1] * L_exact
    rtol = 1e-05
    denom_x = abs(exp_fx) if abs(exp_fx) > 1.0 else 1.0
    denom_y = abs(exp_fy) if abs(exp_fy) > 1.0 else 1.0
    assert abs(sum_fx - exp_fx) <= rtol * denom_x
    assert abs(sum_fy - exp_fy) <= rtol * denom_y
    zero_tol = 1e-12
    non_edge = [i for i in range(8) if i not in edge_nodes]
    for i in non_edge:
        assert abs(r_elem[2 * i]) <= zero_tol
        assert abs(r_elem[2 * i + 1]) <= zero_tol